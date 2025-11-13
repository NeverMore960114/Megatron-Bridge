export CUDA_VISIBLE_DEVICES=0

### Inferencing
# Download T5 weights and VAE weights from "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main"
#   T5: models_t5_umt5-xxl-enc-bf16.pth, google
#   VAE: Wan2.1_VAE.pth

CHECKPOINT_DIR=/opt/megatron_checkpoint_VACE
T5_DIR=~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a
VAE_DIR=~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 examples/recipes/wan/inference_vace.py \
  --model_name vace-1.3B \
  --sizes 832*480 \
  --src_video "test.mp4" \
  --src_mask "src_mask.mp4" \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_step 0000 \
  --t5_checkpoint_dir ${T5_DIR} \
  --vae_checkpoint_dir ${VAE_DIR} \
  --prompts "Two dogs hit each other during boxing." \
  --frame_nums 81 \
  --tensor_parallel_size 1 \
  --context_parallel_size 1 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50