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


from tta.dataset import create_test_dataset, create_test_loader, create_tta_dataset, create_tta_loader
from tta.optim import configure_tta_model, create_tta_optimizer, create_tta_scheduler
from tta.adapt import test_time_adapt_itm
from tta.utils import preprocess_tta_coefficients



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def main(args, config):
    utils.init_distributed_mode(args)
    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
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
    # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
    #     model,
    #     test_loader,
    #     tokenizer,
    #     device,
    #     config
    # )
    # np.save("/data/jiahao/PAB_TTA/debug_embeddings/sims_matrix_t2i.npy", sims_matrix_t2i.detach().cpu().numpy())
    # np.save("/data/jiahao/PAB_TTA/debug_embeddings/image_embeds.npy", image_embeds.detach().cpu().numpy())
    # np.save("/data/jiahao/PAB_TTA/debug_embeddings/text_embeds.npy", text_embeds.detach().cpu().numpy())
    # np.save("/data/jiahao/PAB_TTA/debug_embeddings/text_atts.npy", text_atts.detach().cpu().numpy())
    sims_matrix_t2i = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    image_embeds = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/image_embeds.npy"))#.to(device)
    text_embeds = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/text_embeds.npy"))#.to(device)
    text_atts = torch.from_numpy(np.load("/data/jiahao/PAB_TTA/debug_embeddings/text_atts.npy"))#.to(device)

    # sims_test_result = mAP(sims_matrix_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    # table.add_row([
    #     -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    # ])
    # print("### Zero-Shot ITC Score: ")
    # print(table)
    # labels = test_loader.dataset.g_pids, test_loader.dataset.q_pids #TODO

    # score_test_t2i = evaluation_itm(
    #     model,
    #     device, config, args,
    #     sims_matrix_t2i, image_embeds, text_embeds, text_atts
    # )
    # test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    # table.add_row([
    #     -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    # ])
    # print("### Zero-Shot ITM Score: ")
    # print(table)

    table.add_row([-999, 69.414, 95.197, 97.776, 81.233, 81.233])
    table.add_row([-999, 84.277, 99.039, 99.596, 91.276, 91.276])
    print("### Zero-Shot Score: ")
    print(table)
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
        arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] )
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                model, test_loader, tokenizer, device, config)
            train_stats = test_time_adapt_itm(model, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)#sims_matrix_t2i, image_embeds, text_embeds, text_atts)

            sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
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
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
