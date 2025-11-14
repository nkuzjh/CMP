import os
from pathlib import Path
import time
import datetime
import argparse
import json
import math
import random
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
from prettytable import PrettyTable

import torch
import torch.nn as nn
from typing import List
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from transformers import BertTokenizer

import utils
from models.model_search import Search

# from dataset import create_dataset, create_sampler, create_loader
# from dataset.search_dataset import TextMaskingGenerator
# from scheduler import create_scheduler
# from optim import create_optimizer

# from train import train_model
from eval import evaluation_itm, evaluation_itc, mAP


from tta.dataset import create_test_dataset, create_test_loader, create_tta_dataset, create_tta_loader, create_tta_img_aug_dataset, create_tta_img_aug_loader
from tta.optim import configure_tta_model, create_tta_optimizer, create_tta_scheduler
from tta.adapt import test_time_adapt_itm, test_time_adapt_imgaug_itm, online_test_time_adapt_itm #, test_time_adapt_itm_itc
from tta.utils import preprocess_tta_coefficients

from tta.online_tta.set_tta_model import set_tta_model, freeze_tta_parameters, collect_tta_params, set_tta_optimizer



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def main(args, config):
    # utils.init_distributed_mode(args)
    print('Not using distributed mode')
    args.distributed = False

    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     output_dir:", args.output_dir)

    if args.bs > 0:
        config['batch_size_tta'] = args.bs
    if args.epo > 0:
        config['schedular']['epochs'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['schedular']['epochs'])
    print("     lr:", config['schedular']['lr'])

    print("### Creating test dataset")
    test_dataset = create_test_dataset(config)
    print(f"     test_dataset: {len(test_dataset)}")
    # sample = next(iter(test_dataset))
    # print(sample)

    print("### Creating test dataloader")
    test_loader = create_test_loader(
        [test_dataset],
        batch_size=[config['batch_size_test']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]
    print(f"     test_loader: {len(test_loader)}")

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)
    model = model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))

    print("### Inference ITC similiarity matrix")
    ## 由于使用run.py调用tta.py开启新的子进程，会导致 itc阶段输出的特征 和 itm tta前创建tta_loader输入的特征 被不同进程的device加载，从而产生关于多进程共用cuda的报错；
    ## 因此，进行首次tta前，先运行evaluation_itc和np.save保存itc 特征到本地，后续每次tta实验使用np.load加载即可。
    ## run only at first time to avoid error, then using np.load() to load itm input features.
    sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config
    )
    # sims_matrix_t2i_wo_norm = (image_feats @ text_feats.t()).t()

    # np.save("data/debug_embeddings/q_pids.npy", np.array(test_loader.dataset.q_pids))
    # np.save("data/debug_embeddings/g_pids.npy", np.array(test_loader.dataset.g_pids))
    # np.save("data/debug_embeddings/image_feats.npy", image_feats.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_feats.npy", text_feats.detach().cpu().numpy())
    # np.save("data/debug_embeddings/sims_matrix_t2i_wo_norm.npy", sims_matrix_t2i_wo_norm.detach().cpu().numpy())
    np.save("data/debug_embeddings/sims_matrix_t2i.npy", sims_matrix_t2i.detach().cpu().numpy())
    np.save("data/debug_embeddings/image_embeds.npy", image_embeds.detach().cpu().numpy())
    np.save("data/debug_embeddings/text_embeds.npy", text_embeds.detach().cpu().numpy())
    np.save("data/debug_embeddings/text_atts.npy", text_atts.detach().cpu().numpy())

    # q_pids = torch.from_numpy(np.load("data/debug_embeddings/q_pids.npy"))
    # g_pids = torch.from_numpy(np.load("data/debug_embeddings/g_pids.npy"))
    # image_feats = torch.from_numpy(np.load("data/debug_embeddings/image_feats.npy"))
    # text_feats = torch.from_numpy(np.load("data/debug_embeddings/text_feats.npy"))
    # sims_matrix_t2i_wo_norm = torch.from_numpy(np.load("data/debug_embeddings/sims_matrix_t2i_wo_norm.npy"))
    sims_matrix_t2i = torch.from_numpy(np.load("data/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    image_embeds = torch.from_numpy(np.load("data/debug_embeddings/image_embeds.npy"))#.to(device)
    text_embeds = torch.from_numpy(np.load("data/debug_embeddings/text_embeds.npy"))#.to(device)
    text_atts = torch.from_numpy(np.load("data/debug_embeddings/text_atts.npy"))#.to(device)

    # sims_test_result_wo_norm = mAP(sims_matrix_t2i_wo_norm, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    # table.add_row([
    #     'cos_sim_wo/norm', sims_test_result_wo_norm['R1'], sims_test_result_wo_norm['R5'], sims_test_result_wo_norm['R10'], sims_test_result_wo_norm['mAP'], sims_test_result_wo_norm['mINP']
    # ])
    # print("### Zero-Shot ITC Score wo/norm: ")
    # print(table)
    sims_test_result = mAP(sims_matrix_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    ])
    print("### Zero-Shot ITC Score: ")
    print(table)
    # # labels = test_loader.dataset.g_pids, test_loader.dataset.q_pids #TODO

    # score_test_t2i_wo_norm = evaluation_itm(
    #     model,
    #     device, config, args,
    #     sims_matrix_t2i_wo_norm, image_embeds, text_embeds, text_atts
    # )
    # test_result_wo_norm = mAP(score_test_t2i_wo_norm, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    # table.add_row([
    #     'itm_score_wo/norm', test_result_wo_norm['R1'], test_result_wo_norm['R5'], test_result_wo_norm['R10'], test_result_wo_norm['mAP'], test_result_wo_norm['mINP']
    # ])
    # print("### Zero-Shot ITM Score wo/norm: ")
    # print(table)
    score_test_t2i = evaluation_itm(
        model,
        device, config, args,
        sims_matrix_t2i, image_embeds, text_embeds, text_atts
    )
    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print("### Zero-Shot ITM Score: ")
    print(table)

    # table.add_row(["cos_sim_wo/norm", 58.544, 91.860, 95.703, 73.596, 73.596])
    # table.add_row([-999, 69.414, 95.197, 97.776, 81.233, 81.233])
    # table.add_row(["itm_score_wo/norm", 84.328, 98.989, 99.494, 91.293, 91.293])
    # table.add_row([-999, 84.277, 99.039, 99.596, 91.276, 91.276])
    # print("### Zero-Shot Score: ")
    # print(table)
    ### Zero-Shot ITM Score:
    # +-------------------+--------+--------+--------+--------+--------+
    # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------------------+--------+--------+--------+--------+--------+
    # |  cos_sim_wo/norm  | 58.544 | 91.860 | 95.703 | 73.596 | 73.596 |
    # |        -999       | 69.414 | 95.197 | 97.776 | 81.233 | 81.233 |
    # | itm_score_wo/norm | 84.328 | 98.989 | 99.494 | 91.293 | 91.293 |
    # |        -999       | 84.277 | 99.039 | 99.596 | 91.276 | 91.276 |
    # +-------------------+--------+--------+--------+--------+--------+


    ### pretrained XVLM Zero-Shot ITM Score:
    # +-------+--------+--------+--------+--------+--------+
    # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------+--------+--------+--------+--------+--------+
    # |  -999 | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
    # |  -999 | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
    # +-------+--------+--------+--------+--------+--------+


    if args.tta:
        print("### TTA:")

        # print("### Compute ITC Uncertainty")
        recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list  = preprocess_tta_coefficients(config, sims_matrix_t2i)

        print("### Creating tta dataset")
        tta_dataset = create_tta_dataset(
            config,
            sims_matrix_t2i.cpu(),
            image_embeds.cpu(),
            text_embeds.cpu(),
            text_atts.cpu(),
            recall_types,
            ss_idxs_list,
            uncertaintys_list,
            proba_top1_sim_list,
            proba_inversed_sim_list,

        )
        print(f"     tta_dataset: {len(tta_dataset)}")
        # sample = next(iter(tta_dataset))
        # print(sample)

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[True],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")
        # sample = `next(iter(tta_loader))`
        # print(sample)

        print("### Configure adapted weights")
        # arg_tm = utils.AttrDict(config['tta_model'])
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n,m,m.training) for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([ 1 for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True ]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] ) * config.get('tta_steps', 1)
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(model, test_loader, tokenizer, device, config)
            train_stats = test_time_adapt_itm(model, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)#, sims_matrix_t2i, image_embeds, text_embeds, text_atts)

            if (epoch+1 in [1,2,3,5,10,15,20,30,40,50,60]) or (epoch+1 == max_epoch):
                # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(model, test_loader, tokenizer, device, config)
                score_test_t2i = evaluation_itm(
                    model,
                    device, config, args,
                    sims_matrix_t2i, image_embeds, text_embeds, text_atts
                )

                test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                table.add_row([
                    epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
                ])
                print("### TTA ITM Score: ")
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('     logs: ', logs)

                for k, v in logs.items():
                    logs[k] = str(v)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                result = test_result['R1']
                if result > best:
                    # save_obj = {'model': model.state_dict(), 'config': config, }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = result
                    best_epoch = epoch
                    best_logs = logs

            # del sims_matrix_t2i, image_embeds, text_embeds, text_atts
            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch {best_epoch} : {best_logs}")
        print(f"### best epoch {best_epoch} : {best_logs}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))


def main_img_aug(args, config):
    # utils.init_distributed_mode(args)
    print('Not using distributed mode')
    args.distributed = False
    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     output_dir:", args.output_dir)

    if args.bs > 0:
        config['batch_size_tta'] = args.bs
    if args.epo > 0:
        config['schedular']['epochs'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['schedular']['epochs'])
    print("     lr:", config['schedular']['lr'])


    print("### Creating test dataset")
    test_dataset = create_test_dataset(config)
    print(f"     test_dataset: {len(test_dataset)}")

    print("### Creating test dataloader")
    test_loader = create_test_loader(
        [test_dataset],
        batch_size=[config['batch_size_test']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]
    print(f"     test_loader: {len(test_loader)}")

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)
    model = model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))

    print("### Inference ITC similiarity matrix")
    ## 由于使用run.py调用tta.py开启新的子进程，会导致 itc阶段输出的特征 和 itm tta前创建tta_loader输入的特征 被不同进程的device加载，从而产生关于多进程共用cuda的报错；
    ## 因此，进行首次tta前，先运行evaluation_itc和np.save保存itc 特征到本地，后续每次tta实验使用np.load加载即可。
    ## run only at first time to avoid error, then using np.load() to load itm input features.
    sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config
    )
    # np.save("data/debug_embeddings/sims_matrix_t2i.npy", sims_matrix_t2i.detach().cpu().numpy())
    # np.save("data/debug_embeddings/image_embeds.npy", image_embeds.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_embeds.npy", text_embeds.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_atts.npy", text_atts.detach().cpu().numpy())
    # sims_matrix_t2i = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    # image_embeds = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/image_embeds.npy"))#.to(device)
    # text_embeds = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/text_embeds.npy"))#.to(device)
    # text_atts = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/text_atts.npy"))#.to(device)
    # sims_matrix_t2i = torch.from_numpy(np.load("data/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    # image_embeds = torch.from_numpy(np.load("data/debug_embeddings/image_embeds.npy"))#.to(device)
    # text_embeds = torch.from_numpy(np.load("data/debug_embeddings/text_embeds.npy"))#.to(device)
    # text_atts = torch.from_numpy(np.load("data/debug_embeddings/text_atts.npy"))#.to(device)
    sims_test_result = mAP(sims_matrix_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    ])
    print("### Zero-Shot ITC Score: ")
    print(table)
    # # labels = test_loader.dataset.q_pids #TODO

    score_test_t2i = evaluation_itm(
        model,
        device, config, args,
        sims_matrix_t2i, image_embeds, text_embeds, text_atts
    )
    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print("### Zero-Shot ITM Score: ")
    print(table)

    # table.add_row([-999, 69.414, 95.197, 97.776, 81.233, 81.233])
    # table.add_row([-999, 84.277, 99.039, 99.596, 91.276, 91.276])
    # print("### Zero-Shot Score: ")
    # print(table)
    ### Zero-Shot ITM Score:
    # +-------+--------+--------+--------+--------+--------+
    # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------+--------+--------+--------+--------+--------+
    # |  -999 | 69.414 | 95.197 | 97.776 | 81.233 | 81.233 |
    # |  -999 | 84.277 | 99.039 | 99.596 | 91.276 | 91.276 |
    # +-------+--------+--------+--------+--------+--------+


    if args.tta:
        print("### TTA:")

        # print("### Compute ITC Uncertainty")
        recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list  = preprocess_tta_coefficients(config, sims_matrix_t2i)


        print("### Creating tta dataset")
        tta_dataset = create_tta_dataset(
            config,
            sims_matrix_t2i.cpu(),
            image_embeds.cpu(),
            text_embeds.cpu(),
            text_atts.cpu(),
            recall_types,
            ss_idxs_list,
            uncertaintys_list,
            proba_top1_sim_list,
            proba_inversed_sim_list,

        )
        print(f"     tta_dataset: {len(tta_dataset)}")
        # sample = next(iter(tta_dataset))
        # print(sample)

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[True],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")
        # sample = `next(iter(tta_loader))`
        # print(sample)

        print("### Creating tta img aug dataset")
        tta_img_aug_dataset = create_tta_img_aug_dataset(config)
        print(f"     tta_img_aug_dataset: {len(tta_img_aug_dataset)}")
        # sample = next(iter(tta_img_aug_dataset))
        # print(sample)

        print("### Creating tta img aug dataloader")
        tta_img_aug_loader = create_tta_img_aug_loader(
            [tta_img_aug_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[False],
            collate_fns=[None]
        )[0]
        print(f"     tta_img_aug_loader: {len(tta_img_aug_loader)}")
        # samples = next(iter(tta_img_aug_loader))
        # print(samples)


        print("### Configure adapted weights")
        # arg_tm = utils.AttrDict(config['tta_model'])
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n,m,m.training) for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([ 1 for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True ]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil( len(tta_img_aug_dataset) / config['batch_size_tta'] ) * config.get('tta_steps', 1) # TODO 按理来说这里使用tta_img_aug_dataset的length应当和itc后sims_matrix的length一致，但由于itc后sims_matrix的length是根据ss_idxs_list计算的，因此可能step_per_epoch实际会小于这里的值。
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16


        print("### Start ITC ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
            #     model, test_loader, tokenizer, device, config)
            # train_stats = test_time_adapt_itm_itc(model, tokenizer, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)#sims_matrix_t2i, image_embeds, text_embeds, text_atts)
            train_stats = test_time_adapt_imgaug_itm(model, tokenizer, optimizer, scaler, epoch, device, lr_scheduler, config, tta_img_aug_loader, tta_loader)

            if (epoch+1 in [1,2,3,5,10,15,20,30,40,50,60]) or (epoch+1 == max_epoch):
                sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
                    model, test_loader, tokenizer, device, config)
                score_test_t2i = evaluation_itm(
                    model,
                    device, config, args,
                    sims_matrix_t2i, image_embeds, text_embeds, text_atts
                )

                test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                table.add_row([
                    epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
                ])
                print("### TTA ITM Score: ")
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('     logs: ', logs)

                for k, v in logs.items():
                    logs[k] = str(v)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                result = test_result['R1']
                if result > best:
                    # save_obj = {'model': model.state_dict(), 'config': config, }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = result
                    best_epoch = epoch

            # del sims_matrix_t2i, image_embeds, text_embeds, text_atts
            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)
        print("### best epoch: %d" % best_epoch)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))




def find_meta_modules(model: nn.Module) -> List[str]:
    """
    遍历一个 PyTorch 模型，并返回所有在 'meta' 设备上
    持有参数(parameters)或缓冲区(buffers)的模块名称列表。

    Args:
        model (nn.Module): 要检查的 PyTorch 模型。

    Returns:
        List[str]: 一个包含所有 "meta 模块" 名称的字符串列表。
                   (根模块的名称将是 'root_model')
    """
    meta_module_names = []

    # model.named_modules() 会深度优先遍历所有模块
    # (包括根模块、子模块和子模块的子模块等)
    for name, module in model.named_modules():
        is_meta = False

        # 我们设置 recurse=False，因为 named_modules() 已经在为我们处理递归了。
        # 我们只想检查*直接*属于当前'module'实例的参数和缓冲区。

        # 1. 检查参数 (Parameters)
        for param in module.parameters(recurse=False):
            if param.device.type == 'meta':
                is_meta = True
                break

        if is_meta:
            # 如果根模块 (name == '') 是 meta，我们给它一个更清晰的名字
            meta_module_names.append(name if name else "root_model")
            # 既然已经确认是 meta，就跳过缓冲区的检查，继续下一个模块
            continue

        # 2. 检查缓冲区 (Buffers) - 比如 BatchNorm 的 running_mean
        for buffer in module.buffers(recurse=False):
            if buffer.device.type == 'meta':
                is_meta = True
                break

        if is_meta:
            meta_module_names.append(name if name else "root_model")

    return meta_module_names


def main_online_tta(args, config):
    # utils.init_distributed_mode(args)
    print('Not using distributed mode')
    args.distributed = False

    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     output_dir:", args.output_dir)

    # if args.bs > 0:
    #     config['batch_size_tta'] = args.bs
    # if args.epo > 0:
    #     config['schedular']['epochs'] = args.epo
    # if args.lr > 0:
    #     config['optimizer']['lr'] = args.lr
    #     config['schedular']['lr'] = args.lr
    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['schedular']['epochs'])
    print("     lr:", config['schedular']['lr'])

    print("### Creating test dataset")
    test_dataset = create_test_dataset(config)
    print(f"     test_dataset: {len(test_dataset)}")
    # sample = next(iter(test_dataset))
    # print(sample)

    print("### Creating test dataloader")
    test_loader = create_test_loader(
        [test_dataset],
        batch_size=[config['batch_size_test']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]
    print(f"     test_loader: {len(test_loader)}")

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)

    meta_list = find_meta_modules(model)
    if meta_list:
        print("[诊断] 发现以下模块在 'meta' 设备上:")
        for module_name in meta_list:
            print(f"  - {module_name}")
    else:
        print("[诊断] 所有模块都已在实体设备上。")
    del model.text_encoder.cls.predictions

    model = model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))

    print("### Inference ITC similiarity matrix")
    ## 由于使用run.py调用tta.py开启新的子进程，会导致 itc阶段输出的特征 和 itm tta前创建tta_loader输入的特征 被不同进程的device加载，从而产生关于多进程共用cuda的报错；
    ## 因此，进行首次tta前，先运行evaluation_itc和np.save保存itc 特征到本地，后续每次tta实验使用np.load加载即可。
    ## run only at first time to avoid error, then using np.load() to load itm input features.
    sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config
    )
    sims_matrix_t2i_wo_norm = (image_feats @ text_feats.t()).t()

    # np.save("data/debug_embeddings/q_pids.npy", np.array(test_loader.dataset.q_pids))
    # np.save("data/debug_embeddings/g_pids.npy", np.array(test_loader.dataset.g_pids))
    # np.save("data/debug_embeddings/image_feats.npy", image_feats.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_feats.npy", text_feats.detach().cpu().numpy())
    # np.save("data/debug_embeddings/sims_matrix_t2i_wo_norm.npy", sims_matrix_t2i_wo_norm.detach().cpu().numpy())
    # np.save("data/debug_embeddings/sims_matrix_t2i.npy", sims_matrix_t2i.detach().cpu().numpy())
    # np.save("data/debug_embeddings/image_embeds.npy", image_embeds.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_embeds.npy", text_embeds.detach().cpu().numpy())
    # np.save("data/debug_embeddings/text_atts.npy", text_atts.detach().cpu().numpy())

    # q_pids = torch.from_numpy(np.load("data/debug_embeddings/q_pids.npy"))
    # g_pids = torch.from_numpy(np.load("data/debug_embeddings/g_pids.npy"))
    # image_feats = torch.from_numpy(np.load("data/debug_embeddings/image_feats.npy"))
    # text_feats = torch.from_numpy(np.load("data/debug_embeddings/text_feats.npy"))
    # sims_matrix_t2i_wo_norm = torch.from_numpy(np.load("data/debug_embeddings/sims_matrix_t2i_wo_norm.npy"))
    # sims_matrix_t2i = torch.from_numpy(np.load("data/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    # image_embeds = torch.from_numpy(np.load("data/debug_embeddings/image_embeds.npy"))#.to(device)
    # text_embeds = torch.from_numpy(np.load("data/debug_embeddings/text_embeds.npy"))#.to(device)
    # text_atts = torch.from_numpy(np.load("data/debug_embeddings/text_atts.npy"))#.to(device)

    sims_test_result_wo_norm = mAP(sims_matrix_t2i_wo_norm, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        'cos_sim_wo/norm', sims_test_result_wo_norm['R1'], sims_test_result_wo_norm['R5'], sims_test_result_wo_norm['R10'], sims_test_result_wo_norm['mAP'], sims_test_result_wo_norm['mINP']
    ])
    print("### Zero-Shot ITC Score wo/norm: ")
    print(table)
    sims_test_result = mAP(sims_matrix_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    ])
    print("### Zero-Shot ITC Score: ")
    print(table)
    # labels = test_loader.dataset.g_pids, test_loader.dataset.q_pids #TODO

    score_test_t2i_wo_norm = evaluation_itm(
        model,
        device, config, args,
        sims_matrix_t2i_wo_norm, image_embeds, text_embeds, text_atts
    )
    test_result_wo_norm = mAP(score_test_t2i_wo_norm, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        'itm_score_wo/norm', test_result_wo_norm['R1'], test_result_wo_norm['R5'], test_result_wo_norm['R10'], test_result_wo_norm['mAP'], test_result_wo_norm['mINP']
    ])
    print("### Zero-Shot ITM Score wo/norm: ")
    print(table)
    score_test_t2i = evaluation_itm(
        model,
        device, config, args,
        sims_matrix_t2i, image_embeds, text_embeds, text_atts
    )
    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print("### Zero-Shot ITM Score: ")
    print(table)

    # table.add_row(["cos_sim_wo/norm", 58.544, 91.860, 95.703, 73.596, 73.596])
    # table.add_row([-999, 69.414, 95.197, 97.776, 81.233, 81.233])
    # table.add_row(["itm_score_wo/norm", 84.328, 98.989, 99.494, 91.293, 91.293])
    # table.add_row([-999, 84.277, 99.039, 99.596, 91.276, 91.276])
    # print("### Zero-Shot Score: ")
    # print(table)
    ### Zero-Shot ITM Score:
    # +-------------------+--------+--------+--------+--------+--------+
    # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------------------+--------+--------+--------+--------+--------+
    # |  cos_sim_wo/norm  | 58.544 | 91.860 | 95.703 | 73.596 | 73.596 |
    # |        -999       | 69.414 | 95.197 | 97.776 | 81.233 | 81.233 |
    # | itm_score_wo/norm | 84.328 | 98.989 | 99.494 | 91.293 | 91.293 |
    # |        -999       | 84.277 | 99.039 | 99.596 | 91.276 | 91.276 |
    # +-------------------+--------+--------+--------+--------+--------+


    ### pretrained XVLM Zero-Shot ITM Score:
    # +-------+--------+--------+--------+--------+--------+
    # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------+--------+--------+--------+--------+--------+
    # |  -999 | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
    # |  -999 | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
    # +-------+--------+--------+--------+--------+--------+


    if args.tta:
        print("### TTA:")

        # print("### Compute ITC Uncertainty")
        recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list  = preprocess_tta_coefficients(config, sims_matrix_t2i)

        print("### Creating tta dataset")
        tta_dataset = create_tta_dataset(
            config,
            sims_matrix_t2i.cpu(),
            image_embeds.cpu(),
            text_embeds.cpu(),
            text_atts.cpu(),
            recall_types,
            ss_idxs_list,
            uncertaintys_list,
            proba_top1_sim_list,
            proba_inversed_sim_list,

        )
        print(f"     tta_dataset: {len(tta_dataset)}")
        # sample = next(iter(tta_dataset))
        # print(sample)

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[False],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")
        # sample = next(iter(tta_loader))
        # print(sample)

        print("### Configure adapted weights")
        # arg_tm = utils.AttrDict(config['tta_model'])
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n,m,m.training) for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([ 1 for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True ]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model, device)
        # arg_sche = utils.AttrDict(config['schedular'])
        # arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] ) * config.get('tta_steps', 1)
        lr_scheduler = None#create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        tta_model = set_tta_model(model, optimizer, args)

        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(model, test_loader, tokenizer, device, config)
            train_stats = online_test_time_adapt_itm(args, tta_model, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)#, sims_matrix_t2i, image_embeds, text_embeds, text_atts)

            # if (epoch+1 in [1,2,3,5,10,15,20,30,40,50,60]) or (epoch+1 == max_epoch):
            if (epoch+1 == 1) or (epoch+1 == max_epoch):
                # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(model, test_loader, tokenizer, device, config)
                score_test_t2i = evaluation_itm(
                    tta_model.model,
                    device, config, args,
                    sims_matrix_t2i, image_embeds, text_embeds, text_atts
                )

                test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                table.add_row([
                    epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
                ])
                print("### TTA ITM Score: ")
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('     logs: ', logs)

                for k, v in logs.items():
                    logs[k] = str(v)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                result = test_result['R1']
                if result > best:
                    # save_obj = {'model': model.state_dict(), 'config': config, }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = result
                    best_epoch = epoch
                    best_logs = logs

            # del sims_matrix_t2i, image_embeds, text_embeds, text_atts
            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch {best_epoch} : {best_logs}")
        print(f"### best epoch {best_epoch} : {best_logs}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--method', type=str, default='tcr')

    parser.add_argument('--tta_steps', type=int, default=3)
    parser.add_argument('--con_ratio', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--t', type=float, default=0.1)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if config.get('online_method', False):
        main_online_tta(args, config)
    elif config.get('is_image_augmentation', False):
        main_img_aug(args, config)
    else:
        main(args, config)



# CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs_tta_online_tta/tent.yaml --method tent --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

# CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs_tta_online_tta/tcr.yaml --method tcr --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

# CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs_tta_online_tta/shot.yaml --method shot --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

# CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/sar.yaml --method sar --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

# CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/read.yaml --method read --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta


# # xvlm
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
# +------+--------+--------+--------+--------+--------+
    # # tent max_epoch = 50
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 74.621 | 97.422 | 98.787 | 85.117 | 85.117 |
        # |   49  | 74.115 | 95.956 | 98.180 | 84.179 | 84.179 |
        # +-------+--------+--------+--------+--------+--------+
        ### TTA ITM Score: 3e-4 debug_embedding重新推理
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 74.722 | 97.472 | 98.888 | 85.234 | 85.234 |
        # |         49        | 73.509 | 95.652 | 97.573 | 83.714 | 83.714 |
        # +-------------------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:11
        # Computing matching score time 0:01:24
        ### Time 0:14:24

    # # tcr max_epoch = 50
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 75.430 | 97.624 | 98.837 | 85.653 | 85.653 |
        # |   49  | 73.913 | 94.590 | 96.461 | 83.598 | 83.598 |
        # +-------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:23
        # Computing matching score time 0:02:30
        ### Time 0:37:43

        ### TTA ITM Score:  3e-4 debug_embedding重新推理
        ### TTA ITM Score:
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 75.430 | 98.028 | 98.938 | 85.790 | 85.790 |
        # |         49        | 74.924 | 96.158 | 97.978 | 84.728 | 84.728 |
        # +-------------------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:15
        #  Computing matching score time 0:01:26
        ### Time 0:15:32

    # # shot max_epoch = 50
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 74.924 | 97.472 | 98.837 | 85.335 | 85.335 |
        # |   49  | 73.407 | 95.956 | 97.927 | 83.727 | 83.727 |
        # +-------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:49
        # Computing matching score time 0:04:23
        ### Time 0:50:09

        ### TTA ITM Score:  3e-4 debug_embedding重新推理
        ### TTA ITM Score:
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 74.874 | 97.472 | 98.787 | 85.296 | 85.296 |
        # |         49        | 73.660 | 95.804 | 97.826 | 83.977 | 83.977 |
        # +-------------------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:14
        # Computing matching score time 0:01:23
        ### Time 0:16:47

    # sar max_epoch = 50
        ### TTA ITM Score: lr=3e-4
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 76.593 | 97.219 | 98.382 | 86.183 | 86.183 |
        # |   49  | 76.997 | 97.270 | 98.382 | 86.406 | 86.406 |
        # +-------+--------+--------+--------+--------+--------+
        ### TTA ITM Score: lr=5e-5
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 76.744 | 97.017 | 98.483 | 86.221 | 86.221 |
        # |   49  | 74.823 | 97.219 | 98.686 | 85.278 | 85.278 |
        # +-------+--------+--------+--------+--------+--------+
        ### TTA ITM Score: 5e-5 debug_embedding重新推理
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 72.700 | 97.877 | 99.090 | 84.332 | 84.332 |
        # |         49        | 72.750 | 97.877 | 99.090 | 84.358 | 84.358 |
        # +-------------------+--------+--------+--------+--------+--------+
        ### TTA ITM Score: 3e-4 debug_embedding重新推理
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 73.256 | 97.877 | 99.090 | 84.627 | 84.627 |
        # |         49        | 73.205 | 97.877 | 99.090 | 84.580 | 84.580 |
        # +-------------------+--------+--------+--------+--------+--------+
            #  itm tta time 0:00:24
        #   Computing matching score time 0:01:20
        ### Time 0:23:19

    # read max_epoch = 50
        # ### TTA ITM Score:
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 74.520 | 97.523 | 98.635 | 85.086 | 85.086 |
        # |   49  | 74.671 | 96.714 | 98.231 | 84.857 | 84.857 |
        # +-------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:42
        ### Time 0:43:56
        # Computing matching score time 0:04:21

        ### TTA ITM Score:  lr=5e-5 可能哪里出问题了?
        # +-------+--------+--------+--------+--------+--------+
        # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------+--------+--------+--------+--------+--------+
        # |   0   | 78.665 | 97.422 | 98.332 | 87.422 | 87.422 |
        # |   49  | 80.991 | 97.371 | 98.332 | 88.661 | 88.661 |
        # +-------+--------+--------+--------+--------+--------+


        ### TTA ITM Score: lr=5e-5  debug_embedding重新推理
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 74.216 | 97.674 | 98.938 | 85.108 | 85.108 |
        # |         49        | 74.166 | 96.764 | 98.332 | 84.388 | 84.388 |
        # +-------------------+--------+--------+--------+--------+--------+
        # itm tta time 0:00:12
        # Computing matching score time 0:01:20
        ### Time 0:13:02

        # ### TTA ITM Score:  lr=3e-4  debug_embedding重新推理
        # +-------------------+--------+--------+--------+--------+--------+
        # |       epoch       |   R1   |   R5   |  R10   |  mAP   |  mINP  |
        # +-------------------+--------+--------+--------+--------+--------+
        # |  cos_sim_wo/norm  | 43.023 | 79.828 | 88.726 | 59.082 | 59.082 |
        # |        -999       | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
        # | itm_score_wo/norm | 72.750 | 97.877 | 99.090 | 84.370 | 84.370 |
        # |        -999       | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
        # |         0         | 74.823 | 97.523 | 98.938 | 85.270 | 85.270 |
        # |         49        | 74.621 | 96.006 | 98.180 | 84.611 | 84.611 |
        # +-------------------+--------+--------+--------+--------+--------+
        #   itm tta time 0:00:20
        # Computing matching score time 0:01:41
        #### Time 0:14:30

