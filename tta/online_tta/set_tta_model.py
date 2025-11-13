import torch
from torch import nn
from tta.online_tta.tent import Tent
from tta.online_tta.sar import SAR, SAM
# from online_tta.eata import EATA
from tta.online_tta.read import READ
from tta.online_tta.shot import SHOT
from tta.online_tta.tcr import TCR
# from online_tta.deyo import DeYO
# from online_tta.tcr_untrain import TCR_Untrain

def freeze_tta_parameters(model):

    # model.train()
    # model.requires_grad_(False)
    # print("only_text")
    # for name, param in model.base_model.transformer.named_parameters():
    #     if ('ln' in name):
    #         param.requires_grad_(True)

    # return model

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

def collect_tta_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    # params = []
    # names = []
    # for nm, m in model.base_model.visual.named_modules():
    #     if isinstance(m, (nn.LayerNorm)):
    #         for np, p in m.named_parameters():
    #             if np in ['weight', 'bias']:  # weight is scale, bias is shift
    #                 params.append(p)
    #                 names.append(f"{nm}.{np}")

    # return params, names

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

def set_tta_optimizer(params, config, args):
    if args.method=='sar':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params=params, base_optimizer=base_optimizer, lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(params=params, lr=config['init_lr'], weight_decay=config['weight_decay'])
    return optimizer

def set_tta_model(model, optimizer, args, margin_e0=0):
    if args.method=='tent':
        model=Tent(model, optimizer, steps=args.tta_steps)
    elif args.method=='sar':
        model=SAR(model, optimizer, steps=args.tta_steps, margin_e0=0.40)
    elif args.method=='read':
        model=READ(model, optimizer, steps=args.tta_steps)
    elif args.method=='shot':
        model=SHOT(model, optimizer, steps=args.tta_steps, threshold=0.9, clf_coeff=0.1)
    # elif args.method=='eata':
    #     model=EATA(model, optimizer, steps=args.tta_steps, fishers=None, e_margin=0.40)
    # elif args.method=='deyo':
    #     model=DeYO(model, args, optimizer, steps=args.tta_steps, deyo_margin=0.50, margin_e0=0.40)
    elif args.method=='tcr':
        model=TCR(model, optimizer, steps=args.tta_steps)
    # elif args.method=='tcr_untrain':
    #     model=TCR_Untrain(model, optimizer, steps=1)
    else:
        assert False, "Not correct model name."

    return model

def set_temperature(args, config):
    if args.method == 'tcr':
        return args

    dataset_temperature_map = {
        'flickr': 0.01,
        'coco': 0.01,
        'fashion_gen': 0.001,
        'fashion_gen_detail': 0.001,
        'cuhk_pedes': 0.0001,
        'icfg_pedes': 0.0001,
        'nocaps_entire_domain': 0.01,
        'nocaps_in_domain': 0.01,
        'nocaps_near_domain': 0.01,
        'nocaps_out_domain': 0.01
    }

    dataset = config["dataset"]
    temperature = dataset_temperature_map.get(dataset, 0.01)
    args.temperature = temperature
    return args






