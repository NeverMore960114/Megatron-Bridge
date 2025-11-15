import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

from megatron.bridge.models.wan.flow_matching.flow_inference_pipeline import VACEFlowInferencePipeline
from megatron.bridge.models.wan.inference.configs import SIZE_CONFIGS, SUPPORTED_SIZES, MAX_AREA_CONFIGS, WAN_CONFIGS
from megatron.bridge.models.wan.inference.utils.utils import cache_video, cache_image, str2bool


EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}




def validate_args(args):
    # Basic check
    assert args.checkpoint_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 16
        
    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_nums is None:
        args.frame_nums = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    if args.sizes is not None and len(args.sizes) > 0:
        for s in args.sizes:
            assert s in SUPPORTED_SIZES[args.model_name], f"Unsupport size {s} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    return args


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=None,
        choices=list(SIZE_CONFIGS.keys()),
        help="List of sizes to generate multiple images or videos (WIDTH*HEIGHT). Example: --sizes 1280*720 1920*1080"
    )
    parser.add_argument(
        "--frame_nums",
        type=int,
        nargs="+",
        default=None,
        help="List of frame counts (each should be 4n+1). Broadcasts if single value."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="The path to the main VACE checkpoint directory.")
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help=(
            "Optional training step to load, e.g. 1800 -> iter_0001800. "
            "If not provided, the latest (largest) step in --checkpoint_dir is used.")
    )
    parser.add_argument(
        "--t5_checkpoint_dir",
        type=str,
        default=None,
        help="Optional directory containing T5 checkpoint/tokenizer")
    parser.add_argument(
        "--vae_checkpoint_dir",
        type=str,
        default=None,
        help="Optional directory containing VAE checkpoint")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        nargs="+",
        default=None,
        help="List of name of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        nargs="+",
        default=None,
        help="List of name of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        nargs="+",
        default=None,
        help="List of list of the source reference images. Separated by ','. Default None.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="List of prompt to generate the image or video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size.")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size.")
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel size.")
    parser.add_argument(
        "--sequence_parallel",
        type=str2bool,
        default=False,
        help="Sequence parallel.")

    args = parser.parse_args()

    validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    cfg = WAN_CONFIGS[args.model_name]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompts is None:
        prompts = [None]
    else:
        prompts = args.prompts
        
    if args.src_video is None:
        src_video = [None]
    else:
        src_video = args.src_video
        
    if args.src_mask is None:
        src_mask = [None]
    else:
        src_mask = args.src_mask
        
    if args.src_ref_images is None:
        src_ref_images = [None]
    else:
        src_ref_images = args.src_ref_images

    # Resolve sizes list (default to first supported size for task)
    if args.sizes is not None and len(args.sizes) > 0:
        size_keys = args.sizes
    else:
        size_keys = [SUPPORTED_SIZES[args.model_name][0]]

    # Resolve frame counts list (default 81)
    if args.frame_nums is not None and len(args.frame_nums) > 0:
        frame_nums = args.frame_nums
    else:
        frame_nums = [81]

    # Enforce 1:1 pairing across lists
    assert len(prompts) == len(size_keys) == len(frame_nums), (
        f"prompts ({len(prompts)}), sizes ({len(size_keys)}), and frame_nums ({len(frame_nums)}) "
        f"must have the same length")

    logging.info("Creating VACE flow inference pipeline.")
    pipeline = VACEFlowInferencePipeline(
        config=cfg,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        t5_checkpoint_dir=args.t5_checkpoint_dir,
        vae_checkpoint_dir=args.vae_checkpoint_dir,
        device_id=device,
        rank=rank,
        t5_cpu=args.t5_cpu,
        tensor_parallel_size=args.tensor_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        sequence_parallel=args.sequence_parallel,
        pipeline_dtype=torch.float32,
    )

    # DEBUGGING
    rank = dist.get_rank()
    if rank == 0:
        print("tensor_parallel_size:", args.tensor_parallel_size)
        print("context_parallel_size:", args.context_parallel_size)
        print("pipeline_parallel_size:", args.pipeline_parallel_size)
        print("sequence_parallel:", args.sequence_parallel)
        print("\n\n\n")

    for i in range(len(src_video)):
        sub_src_video, sub_src_mask, sub_src_ref_images = pipeline.prepare_source([src_video[i]],
                                                                  [src_mask[i]],
                                                                  [src_ref_images[i]],
                                                                  frame_nums[i], SIZE_CONFIGS[size_keys[i]], device)
        src_video[i], src_mask[i], src_ref_images[i] = *sub_src_video, *sub_src_mask, *sub_src_ref_images
    
    
    logging.info(
        f"Generating videos ...")
    videos = pipeline.generate(
        prompts=prompts,
        input_frames=src_video,
        input_masks=src_mask,
        input_ref_images=src_ref_images,
        sizes=[SIZE_CONFIGS[size] for size in size_keys],
        frame_nums=frame_nums,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    if rank == 0:
        for i, video in enumerate(videos):
            formatted_experiment_name = (args.save_file) if args.save_file is not None else "DefaultExp"
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompts[i].replace(" ", "_").replace("/",
                                                                    "_")[:50]
            suffix = '.mp4'
            formatted_save_file = f"{args.model_name}_{formatted_experiment_name}_videoindex{int(i)}_size{size_keys[i].replace('*','x') if sys.platform=='win32' else size_keys[i]}_{formatted_prompt}_{formatted_time}" + suffix

            # if "t2v" in args.task:
            logging.info(f"Saving generated video to {formatted_save_file}")
            cache_video(
                tensor=video[None],
                save_file=formatted_save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
                
            cache_video(
                tensor=src_video[i][None],
                save_file=f'{args.model_name}_{formatted_experiment_name}_index{i}_src_video_{formatted_time}.mp4',
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            logging.info(f"Saving src_video to {args.model_name}_{formatted_experiment_name}_index{i}_src_video_{formatted_time}.mp4")

            cache_video(
                tensor=src_mask[i][None],
                save_file=f'{args.model_name}_{formatted_experiment_name}_index{i}_src_mask_{formatted_time}.mp4',
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(0, 1))
            logging.info(f"Saving src_mask to {args.model_name}_{formatted_experiment_name}_index{i}_src_mask_{formatted_time}.mp4")

            if src_ref_images[i] is not None:
                for j, ref_img in enumerate(src_ref_images[i]):
                    cache_image(
                        tensor=ref_img[:, 0, ...],
                        save_file=f'{args.model_name}_{formatted_experiment_name}_index{i}_src_ref_image_{j}_{formatted_time}.png',
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                    logging.info(f"Saving src_ref_image_{j} to {args.model_name}_{formatted_experiment_name}_index{i}_src_ref_image_{j}_{formatted_time}.png")
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
