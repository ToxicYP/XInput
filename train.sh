export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO; export CUDA_LAUNCH_BLOCKING=1;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/conda/envs/bttr/lib
nohup python -u train.py --config config.yaml > log.txt 2>&1 &
# python train.py --config config.yaml

# 改以下文件，模型部分读取
# /root/.local/lib/python3.7/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py00
