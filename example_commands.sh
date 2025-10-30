### Convert checkpoint
See examples/conversion/convert_wan_checkpoints.py for details.


### Finetuning
export HF_TOKEN=...
export WANDB_API_KEY=...
EXP_NAME=...
PRETRAINED_CHECKPOINT=/path/to/pretrained_checkpoint
CHECKPOINT_DIR=/path/to/checkpoint_dir
DATASET_PATH=/path/to/dataset
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=4 examples/recipes/wan/pretrain_wan.py \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.sequence_parallel=false \
  dataset.path=${DATASET_PATH} \
  checkpoint.save=${CHECKPOINT_DIR} \
  checkpoint.load=${PRETRAINED_CHECKPOINT} \
  checkpoint.load_optim=false \
  checkpoint.save_interval=200 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  dataset.global_batch_size=1 \
  dataset.micro_batch_size=1 \
  logger.log_interval=1 \
  logger.wandb_project="wan" \
  logger.wandb_exp_name=${EXP_NAME} \
  logger.wandb_save_dir=${CHECKPOINT_DIR}


### Inferencing
export HF_TOKEN=...
CHECKPOINT_DIR=/path/to/checkpoint_dir
T5_DIR=/path/to/t5_weights
VAE_DIR=/path/to/vae_weights
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=4 examples/recipes/wan/inference_wan.py \
  --task t2v-1.3B \
  --sizes 832*480 \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_step 4000 \
  --t5_checkpoint_dir ${T5_DIR} \
  --vae_checkpoint_dir ${VAE_DIR} \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --frame_nums 81 \
  --tensor_parallel_size 1 \
  --context_parallel_size 4 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50