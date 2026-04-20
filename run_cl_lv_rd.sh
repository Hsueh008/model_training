#!/bin/bash

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NGPU=$SLURM_GPUS_ON_NODE
else
    NGPU=$(nvidia-smi -L | wc -l)
fi

echo "🚀 開始訓練！偵測到 $NGPU 張 GPU..."

SIF_PATH="/work/u8886626/pytorch_env.sif"
MY_PKGS="/work/u8886626/my_packages"
MY_WORK_DIR="/work/u8886626"

singularity exec --nv --bind /work:/work $SIF_PATH bash -c "
    # 使用 PYTHONUSERBASE 完美繼承套件
    export PYTHONUSERBASE=$MY_WORK_DIR/my_packages
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    
    export HF_HOME=$MY_WORK_DIR/hf_cache
    export WANDB_API_KEY='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'
    
    echo '🔍 容器內環境設定完畢，啟動 torchrun...'
    torchrun --nproc_per_node=$NGPU train_cl_lv_rd.py
"