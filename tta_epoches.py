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

from dataset import create_dataset, create_sampler, create_loader
from dataset.search_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from train import train_model
from eval import evaluation_itm, evaluation_itc, mAP
from online_tta.set_tta_model import set_tta_model, freeze_tta_parameters, collect_tta_params, set_tta_optimizer
from online_tta.tta import online_tta_itc#, online_tta_itm



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



def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = utils.get_world_size()

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

    print("### output_dir:", args.output_dir)

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

    print("Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("### Creating search dataset")
    train_dataset, test_dataset = create_dataset(config, True)

    start_time = time.time()

    if args.epochs_tta:
        print("### Start epochs tta")
        model_without_ddp = freeze_tta_parameters(model_without_ddp)
        params = collect_tta_params(model_without_ddp)[0]
        optimizer = set_tta_optimizer(params, config, args=args)
        tta_model = set_tta_model(model_without_ddp, optimizer, args)

        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_tta']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]

        start_time = time.time()
        for epoch in range(config['max_epoch']):
            print(f"\n\n### Epoch: {epoch}")
            sims_matrix_t2i, sims_matrix_t2i_online, image_embeds, text_embeds, text_atts, image_feats, text_feats = online_tta_itc(tta_model, test_loader, tokenizer, device, config, args)

        print("### Finish epochs tta")
        sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(tta_model.model, test_loader, tokenizer, device, config)

        # score_test_t2i = online_tta_itm(tta_model, device, config, args, sims_matrix_t2i, image_embeds, text_embeds, text_atts)
        score_test_t2i = evaluation_itm(tta_model.model, device, config, args, sims_matrix_t2i, image_embeds, text_embeds, text_atts)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Epochs TTA time {}'.format(total_time_str))

        if utils.is_main_process():
            print('evaluating result:')
            mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids)
        dist.barrier()

    elif args.evaluate:
        print("### Start evaluating")
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]
        sims_matrix_t2i, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
            model_without_ddp, test_loader, tokenizer, device, config)
        score_test_t2i = evaluation_itm(model_without_ddp, device, config, args,
                                        sims_matrix_t2i, image_embeds, text_embeds, text_atts)
        if utils.is_main_process():
            print('evaluating result:')
            mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids)
        dist.barrier()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    # parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    # parser.add_argument('--epo', default=0, type=int, help="epoch")
    # parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--epochs_tta', action='store_true')
    parser.add_argument('--method', default='tcr')
    parser.add_argument('--tta_steps', type=int, default=3)
    parser.add_argument('--con_ratio', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--t', type=float, default=0.1)

    args = parser.parse_args()

    args.output_dir = args.output_dir + "/" + args.method

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.config = args.config + "/" + args.method + ".yaml"
    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)


# CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method tcr --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th

# checkpoint/cmp.pth
# checkpoint/16m_base_model_state_step_199999.th



# # xvlm
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
# +------+--------+--------+--------+--------+--------+
    # # tent max_epoch = 50

    # # tcr max_epoch = 50

    # # shot max_epoch = 50

    # sar max_epoch = 50

    # read max_epoch = 50
