nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &

wait


nohup python3 run.py --tta --task "tta_exp0.1"> logs/tta_exp0.1.log 2>&1 &

wait

nohup python3 run.py --tta --task "tta_exp0.2"> logs/tta_exp0.2.log 2>&1 &

wait

nohup python3 run.py --tta --task "tta_exp0.3"> logs/tta_exp0.3.log 2>&1 &

wait

nohup python3 run.py --tta --task "tta_exp0.4"> logs/tta_exp0.4.log 2>&1 &

wait


# # 按顺序执行每个训练任务
# for i in "${!args_files[@]}"
# do
#     # script=${scripts[$i]}
#     log_file=$log_dir/${log_files[$i]}
#     arg_file=${args_files[$i]}
#     # port=${ports[$i]}

#     echo " "
#     echo "Starting Training: $script $arg_file ..." # on port $port
#     start_time=$(date +%s)
#     echo "Start Time: $(date +"%Y-%m-%d %T")"

#     # nohup torchrun --nproc_per_node=$num_gpus --master_port=$port $script > $log_file 2>&1 &
#     # nohup python -m torch.distributed.run --nproc_per_node=$num_gpus --master_port=$port $script --is_tta True --cfg-path $arg_file > $log_file 2>&1 &
#     CUDA_VISIBLE_DEVICES=1 nohup python tta_i2t.py --is_tta True --cfg-path $arg_file > $log_file 2>&1 &

#     # 等待当前任务完成
#     wait

#     # 记录结束时间
#     end_time=$(date +%s)
#     duration=$(( end_time - start_time ))
#     # 格式化时间（分钟和秒）
#     minutes=$(( duration / 60 ))
#     seconds=$(( duration % 60 ))
#     # 结束时间和运行时长
#     echo "End Time: $(date +"%Y-%m-%d %T")"
#     echo "Duration: ${minutes}m${seconds}s"
#     echo "Finsh Training $script $arg_file ."
# done
