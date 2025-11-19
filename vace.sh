export CUDA_VISIBLE_DEVICES=0,1

### Inferencing
# Download T5 weights and VAE weights from "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main"
#   T5: models_t5_umt5-xxl-enc-bf16.pth, google
#   VAE: Wan2.1_VAE.pth

CHECKPOINT_DIR=/opt/megatron_checkpoint_VACE
T5_DIR=/opt/Wan2.1-T2V-1.3B
VAE_DIR=/opt/Wan2.1-T2V-1.3B

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 examples/recipes/wan/inference_vace.py \
  --model_name vace-1.3B \
  --sizes 832*480 \
  --save_file "depth" \
  --src_video "src_video_depth.mp4" \
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