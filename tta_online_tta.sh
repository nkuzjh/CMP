CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/tent.yaml --method tent --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

wait

CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/tcr.yaml --method tcr --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

wait


CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/shot.yaml --method shot --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

wait


CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/sar.yaml --method sar --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

wait


CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs_tta_online_tta/read.yaml --method read --task placeholder --output_dir output_tta_online_tta --checkpoint checkpoint/16m_base_model_state_step_199999.th --seed 42 --tta

wait
