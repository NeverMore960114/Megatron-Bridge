export CUDA_VISIBLE_DEVICES=0,1
export MBRIDGE_PATH=/workspace/vace/Megatron-Bridge
export PYTHONPATH="${MBRIDGE_PATH}/.:${MBRIDGE_PATH}/src/.:/opt/NeMo-Framework-Launcher/launcher_scripts"

### Inferencing
# Download T5 weights and VAE weights from "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main"
#   T5: models_t5_umt5-xxl-enc-bf16.pth, google
#   VAE: Wan2.1_VAE.pth

# CHECKPOINT_DIR=/opt/megatron_checkpoint_VACE
# CHECKPOINT_STEP=0000
CHECKPOINT_DIR="/workspace/checkpoints_vace_ft_I2V"
CHECKPOINT_STEP=1000
T5_DIR="/workspace/checkpoints/T5"
VAE_DIR="/workspace/checkpoints/"

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 examples/recipes/wan/inference_vace.py \
  --model_name vace-1.3B \
  --sizes 832*480 \
  --save_file "test" \
  --src_video "src_video-frameref.mp4" \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_step 1000  \
  --t5_checkpoint_dir ${T5_DIR} \
  --vae_checkpoint_dir ${VAE_DIR} \
  --prompts "Cat jumps from the cabinet." \
  --frame_nums 81 \
  --tensor_parallel_size 1 \
  --context_parallel_size 2 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50

# NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 examples/recipes/wan/inference_vace.py \
#   --model_name vace-1.3B \
#   --sizes 832*480 832*480 832*480 \
#   --save_file "test" \
#   --src_video "src_video_depth.mp4" "src_video_flow.mp4" "src_video_pose.mp4" \
#   --checkpoint_dir ${CHECKPOINT_DIR} \
#   --checkpoint_step 0000 \
#   --t5_checkpoint_dir ${T5_DIR} \
#   --vae_checkpoint_dir ${VAE_DIR} \
#   --prompts "Two dogs hit each other during boxing." "Two dogs hit each other during boxing." "Two dogs hit each other during boxing." \
#   --frame_nums 81 81 81 \
#   --tensor_parallel_size 1 \
#   --context_parallel_size 2 \
#   --pipeline_parallel_size 1 \
#   --sequence_parallel False \
#   --base_seed 42 \
#   --sample_steps 50