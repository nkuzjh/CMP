
CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method tent --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --tta_steps 3
wait


CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method tcr --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --tta_steps 3
wait



CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method shot --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --tta_steps 3
wait

CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method sar --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --tta_steps 3
wait

CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=10000 tta_epoches.py --epochs_tta --seed 42 --method read --config configs_epochs_tta --output_dir output_epochs_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --tta_steps 3
wait