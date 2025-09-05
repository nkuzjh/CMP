#!/bin/bash

# 设置基础参数
# num_gpus=1
log_dir="./logs"
mkdir -p $log_dir


# 定义日志文件名
log_files=(
    "tta_exp10.log"
    "tta_exp10.1.log"
    "tta_exp10.2.log"
    "tta_exp10.3.log"
    "tta_exp10.4.log"
    "tta_exp10.5.log"
    "tta_exp10.6.log"
)

# 定义参数文件名
task_names=(
    "tta_exp10"
    "tta_exp10.1"
    "tta_exp10.2"
    "tta_exp10.3"
    "tta_exp10.4"
    "tta_exp10.5"
)

# 按顺序执行每个训练任务
for i in "${!task_names[@]}"
do
    # script=${scripts[$i]}
    log_file=$log_dir/${log_files[$i]}
    task_name=${task_names[$i]}
    # port=${ports[$i]}

    echo " "
    echo "Starting Training: $task_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"

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
