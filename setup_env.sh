#运行 source setup_env.sh 设置环境变量
#!/bin/bash
# 设置 CUDA 和 CUPTI 环境变量
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH

# 激活 conda 环境
source /home/ril/anaconda3/etc/profile.d/conda.sh
conda activate limx_walk1_env
export ROBOT_TYPE=SF_TRON1A_ARM

echo "环境设置完成！"
echo "CUDA_HOME: $CUDA_HOME"
echo "现在可以运行训练脚本了"
