#!/bin/bash
# VACE Finetuning Script
# This script demonstrates how to finetune the VACE video editing model

export MBRIDGE_PATH=/workspace/vace/Megatron-Bridge
export PYTHONPATH="${MBRIDGE_PATH}/.:${MBRIDGE_PATH}/src/.:/opt/NeMo-Framework-Launcher/launcher_scripts"

### Prepare energon dataset
python src/megatron/bridge/data/wan/prepare_energon_dataset_vace.py \
  --video_dir /workspace/all_mixkit_segmented \
  --output_dir /workspace/all_mixkit_energon_vace_V2V \
  --checkpoint_dir /opt/megatron_checkpoint_VACE \
  --t5_checkpoint_dir /workspace/checkpoints/T5 \
  --vae_checkpoint_dir /workspace/checkpoints/ \
  --vace_mode V2V \
  --device cuda \
  --height 224 --width 224 --resize_mode bilinear --center-crop \
  --shard_maxcount 100 2>&1 | tee /tmp/prepare_log.txt

energon prepare /workspace/all_mixkit_energon_vace_V2V

# ============================
# Configuration Parameters
# ============================

DATASET_PATH="/workspace/all_mixkit_energon_vace_V2V"
PRETRAINED_CHECKPOINT="/opt/megatron_checkpoint_VACE"
CHECKPOINT_DIR="/workspace/checkpoints_vace_ft_V2V"
EXP_NAME=wan_vace_ft_V2V

# ============================
# Launch Training
# ============================

echo "Starting VACE finetuning..."
echo ""

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=2 examples/recipes/wan/pretrain_vace.py \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.context_parallel_size=1 \
    model.sequence_parallel=false \
    model.qkv_format=thd \
    dataset.path=${DATASET_PATH} \
    dataset.num_workers=0 \
    checkpoint.save=${CHECKPOINT_DIR} \
    checkpoint.load=${PRETRAINED_CHECKPOINT} \
    checkpoint.load_optim=false \
    checkpoint.save_interval=500 \
    optimizer.lr=5e-6 \
    optimizer.min_lr=5e-6 \
    train.eval_iters=0 \
    scheduler.lr_decay_style=constant \
    scheduler.lr_warmup_iters=0 \
    model.seq_length=512 \
    dataset.seq_length=512 \
    train.global_batch_size=2 \
    train.micro_batch_size=1 \
    dataset.global_batch_size=2 \
    dataset.micro_batch_size=1 \
    logger.log_interval=1 \
    logger.wandb_project="vace" \
    logger.wandb_exp_name=${EXP_NAME} \
    logger.wandb_save_dir=${CHECKPOINT_DIR}
    # train.train_iters=$TRAIN_ITERS \
    # train.eval_interval=$EVAL_INTERVAL \
echo ""
echo "=========================================="
echo "VACE Finetuning Complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "=========================================="
