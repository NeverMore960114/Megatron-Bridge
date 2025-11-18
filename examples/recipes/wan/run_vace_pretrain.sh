#!/bin/bash
# VACE Finetuning Script
# This script demonstrates how to finetune the VACE video editing model

# Exit on error
set -e

# ============================
# Configuration Parameters
# ============================

# Dataset path - Update this to point to your energon dataset
DATASET_PATH="${DATASET_PATH:-/workspace/all_mixkit_energon}"

# Checkpoint directories
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/workspace/checkpoints/megatron_checkpoint_1.3B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_ft}"

# Experiment name
EXP_NAME="${EXP_NAME:-vace_mixkit_finetune}"

# Model parallelism settings
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
CONTEXT_PARALLEL="${CONTEXT_PARALLEL:-1}"

# Training hyperparameters
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-5e-6}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-24}"

# Training iterations and intervals
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_ITERS="${EVAL_ITERS:-0}"

# Number of GPUs
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# ============================
# Validation
# ============================

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    echo "Please set DATASET_PATH environment variable or update the script"
    exit 1
fi

# Check if pretrained checkpoint exists
if [ ! -d "$PRETRAINED_CHECKPOINT" ]; then
    echo "Warning: Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT"
    echo "Will start training from scratch or use checkpoint from CHECKPOINT_DIR if available"
fi

# ============================
# Environment Setup
# ============================

echo "=========================================="
echo "VACE Finetuning Configuration"
echo "=========================================="
echo "Dataset Path: $DATASET_PATH"
echo "Pretrained Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output Checkpoint Dir: $CHECKPOINT_DIR"
echo "Experiment Name: $EXP_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Pipeline Parallel: $PIPELINE_PARALLEL"
echo "Context Parallel: $CONTEXT_PARALLEL"
echo "Learning Rate: $LEARNING_RATE"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Number of GPUs: $NPROC_PER_NODE"
echo "=========================================="
echo ""

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# ============================
# Launch Training
# ============================

# Enable fused attention for better performance
export NVTE_FUSED_ATTN=1

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting VACE finetuning..."
echo ""

torchrun --nproc_per_node=$NPROC_PER_NODE \
    "$SCRIPT_DIR/pretrain_vace.py" \
    model.tensor_model_parallel_size=$TENSOR_PARALLEL \
    model.pipeline_model_parallel_size=$PIPELINE_PARALLEL \
    model.context_parallel_size=$CONTEXT_PARALLEL \
    model.sequence_parallel=false \
    model.qkv_format=thd \
    dataset.path="$DATASET_PATH" \
    checkpoint.save="$CHECKPOINT_DIR" \
    checkpoint.load="$PRETRAINED_CHECKPOINT" \
    checkpoint.load_optim=false \
    checkpoint.save_interval=$SAVE_INTERVAL \
    optimizer.lr=$LEARNING_RATE \
    optimizer.min_lr=$MIN_LEARNING_RATE \
    train.eval_iters=$EVAL_ITERS \
    train.eval_interval=$EVAL_INTERVAL \
    scheduler.lr_decay_style=constant \
    scheduler.lr_warmup_iters=0 \
    model.seq_length=$SEQ_LENGTH \
    dataset.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    dataset.global_batch_size=$GLOBAL_BATCH_SIZE \
    dataset.micro_batch_size=$MICRO_BATCH_SIZE \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project="vace" \
    logger.wandb_exp_name="$EXP_NAME" \
    logger.wandb_save_dir="$CHECKPOINT_DIR"

echo ""
echo "=========================================="
echo "VACE Finetuning Complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "=========================================="
