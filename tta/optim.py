from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR


from torch import nn


import torch
from tta.online_tta.sar import SAR, SAM



def configure_model_xvlm_itm(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.text_encoder.modules(): # 针对blip模型结构，仅tta更新Qformer参数；freeze visual_encoder/query_tokens/temp(erature)/image_proj/text_proj/itm_head的参数；
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    for text_encoder_layer_index in range(0,6): # 针对cmp_xvlm模型关闭bert前6层仅用作text_encoder的梯度更新功能
        text_encoder_layer = model.text_encoder.bert.encoder.layer[text_encoder_layer_index]
        for m in text_encoder_layer.modules(): # 针对blip模型结构，仅tta更新Qformer参数；freeze visual_encoder/query_tokens/temp(erature)/image_proj/text_proj/itm_head的参数；
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(False)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    for m in model.itm_head.modules(): # tta更新itm_head参数；freeze others: visual_encoder/query_tokens/temp(erature)/image_proj/text_proj
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_params_xvlm_itm(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.text_encoder.named_modules():

        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    for nm, m in model.itm_head.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_tta_model(config, model):
    model = configure_model_xvlm_itm(model)
    if config.get("uncertainty_temper_is_learnable", False) == True:
        model.uncertainty_temper.requires_grad_(True)
    if config.get("is_prompt_learning", False) == True:
        model.prompt_learning_embedding.requires_grad_(True)
        model.is_prompt_learning=True
    # params, param_names = collect_params_xvlm_itm(model)
    # if getattr(cfg.config.tta, "coeffi_exp_temper_is_learnable", False) == True:
    #     params.append(model.coeffi_exp_temper)
    # if getattr(cfg.config.tta, "is_prompt_learning", False) == True:
    #     params.append(model.learnable_empty_embedding)
    return model


def create_tta_optimizer(args, model, device):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print("     lr: ", lr, "   lr_mult: ", lr_mult, flush=True)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult},
    ]

    no_decay = {"bias",
                "LayerNorm.bias",
                "LayerNorm.weight",
                "norm.bias",
                "norm.weight",
                "norm1.bias",
                "norm1.weight",
                "norm2.bias",
                "norm2.weight"}

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("     model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            if n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
        else:  # decay
            if n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)

    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    if args.method=='sar':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(device=device, params=optimizer_grouped_parameters, base_optimizer=base_optimizer, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    else:
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))


    return optimizer


def create_tta_scheduler(args, optimizer):
    if 'num_tta_steps' not in args:
        args['num_tta_steps'] = args['epochs'] * args['step_per_epoch']
    print("     num_tta_steps: ", args['num_tta_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_tta_steps'] * args['num_warmup_steps'])
    print("     num_warmup_steps: ", args['num_warmup_steps'], flush=True)

    print('     sched:', args.sched, flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_tta_steps - current_step) / float(
                    max(1, args.num_tta_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    elif args.sched == 'step':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            elif current_step < args.num_warmup_steps * 4:
                tt = 1
            elif current_step < args.num_warmup_steps * 7:
                tt = 0.5
            else:
                tt = 0.2

            return tt * max(
                0.0, float(args.num_tta_steps - current_step) / float(
                    max(1, args.num_tta_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
