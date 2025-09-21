
# For the first time you use wordnet
# import nltk
# nltk.download('wordnet')

import os
import argparse

from datetime import datetime


# Set it correctly for distributed training across nodes
NNODES = 1
NODE_RANK = 0
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 1612  # 0~65536
NPROC_PER_NODE = 2 # 4  # e.g. 4 gpus

print("NNODES, ", NNODES)
print("NODE_RANK, ", NODE_RANK)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)


def get_dist_launch(args):
    if args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.run --nproc_per_node=4 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 python3 -m torch.distributed.run --nproc_per_node=2 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.run --nproc_per_node=1 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(num, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    else:
        raise ValueError


def run(args):
    if 'base' in args.task or 'cmp' in args.task:
        args.config = 'configs/' + args.task + '.yaml'
        print(args.task, args.config)

        dist_launch = get_dist_launch(args)
        os.system(
            f"{dist_launch} Search.py --config {args.config} --task {args.task} --output_dir {args.output_dir} "
            f"--checkpoint {args.checkpoint} --bs {args.bs} --epo {args.epo} --lr {args.lr} --seed {args.seed} "
            f"{'--evaluate' if args.evaluate else ''}")

    elif 'tta' in args.task or 'exp' in args.task:
        args.config = 'configs_rerun4/' + args.task + '.yaml'
        args.output_dir = 'output_rerun4/' + args.task + f'/{datetime.now().strftime("%Y%m%d%H%M%S")[:-1]}'
        print("task: ", args.task)
        print("config: ", args.config)
        print("output_dir: ", args.output_dir)

        os_command_str = f"python3 tta.py --config {args.config} --task {args.task} --output_dir {args.output_dir} --checkpoint {args.checkpoint} --bs {args.bs} --epo {args.epo} --lr {args.lr} --seed {args.seed} {'--tta' if args.tta else ''}"

        print(os_command_str)
        # CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_debug.yaml --task tta_debug --output_dir output/tta_debug/2025081715503  --checkpoint checkpoint/cmp.pth --bs 1 --epo 10 --lr 0.0001 --seed 42  --tta

        os.system(os_command_str)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cmp', type=str)
    parser.add_argument('--dist', default='f4', type=str, help="see func get_dist_launch for details")
    parser.add_argument('--output_dir', default='out/cmp', type=str, help='local path')
    parser.add_argument('--checkpoint', default='checkpoint/cmp.pth', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true', help="directly evaluation")
    parser.add_argument('--tta', action='store_true', help="run test time adaptation & evaluation")
    args = parser.parse_args()

    run(args)
