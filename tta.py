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

from dataset import create_dataset, create_sampler, create_loader
from dataset.search_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from train import train_model
from eval import evaluation_itm, evaluation_itc, mAP


from tta.dataset import create_tta_dataset, create_tta_loader
from tta.optim import configure_tta_model, create_tta_optimizer, create_tta_scheduler
from tta.adapt import test_time_adapt_itm



def main(args, config):
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

    print("### Creating tta dataset")
    tta_dataset, test_dataset = create_tta_dataset(config, args.tta)
    print(f"     dataset: {len(tta_dataset)}")

    print("### Creating tta dataloader")
    tta_loader, test_loader = create_tta_loader(
        [tta_dataset, test_dataset],
        batch_size=[config['batch_size_tta'], config['batch_size_test']],
        num_workers=[4, 4],
        is_trains=[True, False],
        collate_fns=[None, None]
    )
    print(f"     tta_loader: {len(tta_loader)}, test_loader: {len(test_loader)}")

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)
    model = model.to(device)
    print("     Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("### Inference ITC similiarity matrix")
    sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config
    )

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    if args.tta:
        print("### TTA:")

        # print("### Compute ITC Uncertainty")
        # TODO uncertaintys, ss_idxs = compute_uncertainty_itc(sims_matrix_t2i, image_embeds, text_embeds, text_atts)

        print("### Configure adapted weights")
        arg_tm = utils.AttrDict(config['tta_model'])
        model = configure_tta_model(arg_tm, model)
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil( tta_dataset_size / config['batch_size_tta'] )
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                              config['skipgram_prb'], config['skipgram_size'],
                                              config['mask_whole_word'])

        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            # train_stats = test_time_adapt_itm(model, tta_loader, optimizer, scaler, tokenizer, epoch, device, lr_scheduler, config, mask_generator)
            train_stats = test_time_adapt_itm(model, optimizer, scaler, epoch, device, lr_scheduler, config, sims_matrix_t2i, image_embeds, text_embeds, text_atts)


            # sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
            #     model, test_loader, tokenizer, device, config)
            score_test_t2i = evaluation_itm(
                model,
                device, config, args,
                sims_matrix_t2i, image_embeds, text_embeds, text_atts
            )

            test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
            table.add_row([
                epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
            ])
            print("     ", table)

            logs = {'epo': epoch}
            for k, v in test_result.items():
                logs[k] = np.around(v, 3)
            for k, v in train_stats.items():
                logs[k] = float(v)
            print('     logs', logs)

            for k, v in logs.items():
                logs[k] = str(v)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(logs) + "\n")

            result = test_result['R1']
            if result > best:
                save_obj = {'model': model.state_dict(), 'config': config, }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
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
