# Bash语法：
# 临时加入环境变量
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
mamba activate .core
python -m bin.run_patchcore


#!/bin/bash

# 定义日志文件路径
LOG_FILE="log$(date '+%Y-%m-%d_%H-%M-%S').txt"

# 写入日志头（带时间戳）
echo -e "\n=====================================" >> "$LOG_FILE"
echo "开始执行时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo -e "=====================================\n" >> "$LOG_FILE"

# 定义需要运行的数据集名称列表
DATASETS=(
    "adm"
    "biggan"
    "glide"
    "midjourney"
    "sdv5"
    "vqdm"
    "wukong"
)

# 循环遍历每个数据集名称并执行命令
for DATASET in "${DATASETS[@]}"; do
    # 终端和日志同时输出进度信息
    PROGRESS_MSG="\n=====================================\n开始运行数据集: $DATASET\n=====================================\n"
    echo -e "$PROGRESS_MSG" | tee -a "$LOG_FILE"
    
    # 执行python命令，输出同时写入日志和终端
    python -m bin.run_patchcore --dataset_names "$DATASET" 2>&1 | tee -a "$LOG_FILE"
    
    # 检查命令执行状态并记录
    if [ $? -eq 0 ]; then
        SUCCESS_MSG=" 数据集 $DATASET 运行完成"
    else
        SUCCESS_MSG=" 数据集 $DATASET 运行失败"
    fi
    echo -e "$SUCCESS_MSG\n" | tee -a "$LOG_FILE"
done

# 写入执行结束信息
echo -e "\n=====================================" >> "$LOG_FILE"
echo "所有数据集执行结束: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo -e "=====================================\n" >> "$LOG_FILE"

echo -e "\n 所有数据集运行完毕！日志已保存至: $LOG_FILE"

############# Detection
### IM224:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC: 0.992, Pixelwise AUROC: 0.981, PRO: 0.944
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model 
--log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu 
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 
--anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset 
--resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
