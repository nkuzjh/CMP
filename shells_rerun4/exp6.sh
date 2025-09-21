#!/bin/bash

# 设置基础参数
# num_gpus=1
log_dir="./logs_rerun4"
mkdir -p $log_dir

# 定义参数文件名
task_names=(
    "exp6"
    "exp6.0.1"
    "exp6.0.2"
    "exp6.0.3"
    "exp6.0.4"
    # "exp6.1"
    # "exp6.1.1"
    # "exp6.1.2"
    # "exp6.1.3"
    # "exp6.1.4"
)

# 按顺序执行每个训练任务
for i in "${!task_names[@]}"
do
    # script=${scripts[$i]}
    log_file=$log_dir/${task_names[$i]}.log
    task_name=${task_names[$i]}
    # port=${ports[$i]}

    echo " "
    echo "Starting Training: $task_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"

    # CUDA_VISIBLE_DEVICES=1
    nohup python3 run.py --tta --task $task_name > $log_file 2>&1 &

    # 等待当前任务完成
    wait

    # 记录结束时间
    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    # 格式化时间（分钟和秒）
    minutes=$(( duration / 60 ))
    seconds=$(( duration % 60 ))
    # 结束时间和运行时长
    echo "End Time: $(date +"%Y-%m-%d %T")"
    echo "Duration: ${minutes}m${seconds}s"
    echo "Finsh Training $task_name ."
done
