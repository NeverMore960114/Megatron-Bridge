import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import webdataset as wds
import cv2
import numpy as np
from tqdm import tqdm

from megatron.bridge.models.wan.flow_matching.flow_inference_pipeline import VACEFlowInferencePipeline
from megatron.bridge.models.wan.inference.configs import WAN_CONFIGS
from megatron.bridge.models.wan.utils.utils import patchify
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

def _map_interpolation(resize_mode: str) -> int:
    interpolation_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if resize_mode not in interpolation_map:
        raise ValueError(f"Invalid resize_mode '{resize_mode}'. Choose from: {list(interpolation_map.keys())}")
    return interpolation_map[resize_mode]


def _calculate_resize_dimensions(
    original_height: int,
    original_width: int,
    target_size: Optional[Tuple[int, int]],
    maintain_aspect_ratio: bool,
) -> Tuple[int, int]:
    if target_size is None:
        return original_height, original_width

    target_height, target_width = target_size
    if not maintain_aspect_ratio:
        return target_height, target_width

    original_aspect = original_width / max(1, original_height)
    target_aspect = target_width / max(1, target_height)

    if original_aspect > target_aspect:
        new_width = target_width
        new_height = int(round(target_width / max(1e-6, original_aspect)))
    else:
        new_height = target_height
        new_width = int(round(target_height * original_aspect))

    return new_height, new_width


def _resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> np.ndarray:
    if target_size is None:
        return frame

    original_height, original_width = frame.shape[:2]
    resize_height, resize_width = _calculate_resize_dimensions(
        original_height, original_width, target_size, maintain_aspect_ratio
    )

    interpolation = _map_interpolation(resize_mode)
    resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

    if maintain_aspect_ratio and center_crop:
        target_height, target_width = target_size
        if resize_height != target_height or resize_width != target_width:
            y_start = max(0, (resize_height - target_height) // 2)
            x_start = max(0, (resize_width - target_width) // 2)
            y_end = min(resize_height, y_start + target_height)
            x_end = min(resize_width, x_start + target_width)
            resized_frame = resized_frame[y_start:y_end, x_start:x_end]

            if resized_frame.shape[0] < target_height or resized_frame.shape[1] < target_width:
                pad_height = max(0, target_height - resized_frame.shape[0])
                pad_width = max(0, target_width - resized_frame.shape[1])
                # Handle both 2D (grayscale/mask) and 3D (RGB) frames
                if resized_frame.ndim == 2:
                    pad_spec = ((0, pad_height), (0, pad_width))
                else:
                    pad_spec = ((0, pad_height), (0, pad_width), (0, 0))
                resized_frame = np.pad(
                    resized_frame, pad_spec, mode="constant", constant_values=0
                )

    return resized_frame

def _read_sidecar_caption(jsonl_path: Path) -> str:
    if not jsonl_path.exists():
        return ""
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Prefer keys used across datasets
                for key in ("vila_caption", "gemini_v2_caption", "caption", "text"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
                # If no known key, try first string value
                for v in obj.values():
                    if isinstance(v, str):
                        return v
                break
    except Exception:
        return ""
    return ""


def _get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, total)


def _load_metadata(video_folder: Path) -> List[Dict]:
    meta_path = video_folder / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)

    # Fallback: scan for .mp4 files with sidecar .jsonl; use full frame range
    items: List[Dict] = []
    for entry in sorted(video_folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".mp4":
            continue
        video_name = entry.name
        video_path = str(entry)
        total_frames = _get_total_frames(video_path)
        start_frame = 0
        end_frame = max(0, total_frames - 1)
        sidecar = entry.with_suffix("")
        # Handle names with additional dots gracefully
        sidecar_jsonl = Path(str(entry).rsplit(".", 1)[0] + ".jsonl")
        caption = _read_sidecar_caption(sidecar_jsonl)
        items.append(
            {
                "file_name": video_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "vila_caption": caption,
            }
        )
    if not items:
        raise FileNotFoundError(f"No meta.json and no .mp4 files found in {video_folder}")
    return items

def _load_frames_cv2(
    video_path: str,
    start_frame: int,
    end_frame: int,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
    target_dtype: torch.dtype,
    is_mask: bool = False,
) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if is_mask:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_frame(frame, target_size, resize_mode, maintain_aspect_ratio, center_crop)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    video_array = np.array(frames)  # T, H, W, C (RGB) or T, H, W (mask) in [0,1]
    video_tensor = torch.from_numpy(video_array)
    
    if is_mask:
        # For masks: T, H, W -> 1, 1, T, H, W
        video_tensor = video_tensor.unsqueeze(0).unsqueeze(0)  # 1, 1, T, H, W
    else:
        # For RGB: T, H, W, C -> 1, C, T, H, W
        video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # 1, C, T, H, W
    
    video_tensor = video_tensor.to(dtype=target_dtype)
    return video_tensor


@torch.no_grad()
def _encode_video_latents(
    vae: AutoencoderKLWan,
    device: str,
    video_tensor: torch.Tensor,
    # deterministic_latents: bool,
) -> torch.Tensor:
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
    video_tensor = video_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    latent_dist = vae.encode(video_tensor)
    # if deterministic_latents:
    #     video_latents = latent_dist[0].mean
    # else:
    #     video_latents = latent_dist[0].sample()
    video_latents = latent_dist[0]

    latent_mean = video_latents.mean().item()
    latent_std = video_latents.std().item()

    if abs(latent_mean) < 0.5 and 0.5 < latent_std < 2.0:
        final_latents = video_latents
    else:
        if not hasattr(vae.config, "latents_mean") or not hasattr(vae.config, "latents_std"):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        final_latents = (video_latents - latents_mean) / latents_std

    return final_latents

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare VACE WebDataset shards using VACEFlowInferencePipeline")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing *_src_video.mp4 and *_mask.mp4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write webdataset shards")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="VACE checkpoint directory")
    parser.add_argument("--checkpoint_step", type=int, default=0000, help="Checkpoint step (optional)")
    parser.add_argument("--vae_checkpoint_dir", type=str, default=None, help="VAE checkpoint directory (optional)")
    parser.add_argument("--t5_checkpoint_dir", type=str, default=None, help="T5 checkpoint directory (optional)")
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="Max samples per shard")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the pipeline on")
    parser.add_argument("--height", type=int, default=None, help="Target height for resizing frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for resizing frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--center-crop", action="store_true", help="Center crop to exact target size after resize")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic latents from VAE encoder")
    parser.add_argument("--model_name", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()), help="The model name to run.")
    parser.add_argument("--vace_mode", default="T2V", choices=["T2V", "I2V", "V2V"], help="VACE mode: T2V, I2V or V2V")
    args = parser.parse_args()

    video_folder = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / "shard-%06d.tar")
    model_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    # Target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif (args.height is None) ^ (args.width is None):
        parser.error("Both --height and --width must be specified together")

    cfg = WAN_CONFIGS[args.model_name]
    pipeline = VACEFlowInferencePipeline(
        config=cfg,  # You may need to load config as in your training/inference scripts
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        t5_checkpoint_dir=args.t5_checkpoint_dir,
        vae_checkpoint_dir=args.vae_checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
    )
    pipeline.text_encoder.model.to(pipeline.device)
    # Load metadata list
    metadata_list = _load_metadata(video_folder)
    with wds.ShardWriter(shard_pattern, maxcount=args.shard_maxcount) as sink:
        written = 0
        for idx, meta in enumerate(tqdm(metadata_list)):
            video_name = meta["path"]
            start_frame = int(meta['frame_idx'].split(':')[0])  # inclusive
            end_frame = int(meta['frame_idx'].split(':')[1])      # inclusive
            prompt = meta["cap"]
            
            video_path = os.path.join(args.video_dir, video_name)
            video_base = os.path.split(video_path)[0]
            src_video_path = os.path.join(video_base, "src_video_obj_1.mp4")
            mask_path = os.path.join(video_base, "mask_obj_1.mp4")
            
            video_tensor = _load_frames_cv2(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                target_size=target_size,
                resize_mode=args.resize_mode,
                maintain_aspect_ratio=not args.no_aspect_ratio,
                center_crop=args.center_crop,
                target_dtype=model_dtype,
            )
            T, H, W = video_tensor.shape[2:5]
            if not os.path.exists(src_video_path) or not os.path.exists(mask_path):
                if args.vace_mode == "T2V":
                    src_video_tensor = torch.zeros((1, 3, T, H, W), device=pipeline.device)
                    mask_tensor = torch.ones((1, 1, T, H, W), device=pipeline.device).div(255.0)
                elif args.vace_mode == "I2V":
                    #Read first frame from video as src_video and remaining frames as zeros
                    src_video_tensor = torch.zeros((3, T, H, W), device=video_tensor.device, dtype=video_tensor.dtype)
                    src_video_tensor[:, 0] = video_tensor[0, :, 0, :, :] # C, T, H, W
                    src_video_tensor = src_video_tensor.unsqueeze(0).to(pipeline.device)  # 1, C, T, H, W
                    mask_tensor = torch.ones((1, 1, T, H, W), device=pipeline.device).div(255.0)    # 1, 1, T, H, W
                elif args.vace_mode == "V2V":
                    print(f"Failed to context read frames for {src_video_path}")
                    continue
            else:
                src_video_tensor = _load_frames_cv2(src_video_path,
                                                    start_frame=start_frame,
                                                    end_frame=end_frame,
                                                    target_size=target_size,
                                                    resize_mode=args.resize_mode,
                                                    maintain_aspect_ratio=not args.no_aspect_ratio,
                                                    center_crop=args.center_crop,
                                                    target_dtype=model_dtype,
                                                    ).to(pipeline.device) 
                mask_tensor = _load_frames_cv2(mask_path,
                                              start_frame=start_frame,
                                              end_frame=end_frame,
                                              target_size=target_size,
                                              resize_mode=args.resize_mode,
                                              maintain_aspect_ratio=not args.no_aspect_ratio,
                                              center_crop=args.center_crop,
                                              target_dtype=model_dtype,
                                              is_mask=True
                                              ).to(pipeline.device) 
            
            # Use pipeline to encode frames/masks and get vace_context
            text_embed = pipeline.text_encoder([prompt], pipeline.device)[0]
            latents = _encode_video_latents(
                vae=pipeline.vae,
                device=pipeline.device,
                video_tensor=video_tensor,
                # deterministic_latents=not args.stochastic,
            )
            vace_context0 = pipeline.vace_encode_frames(src_video_tensor, ref_images=None, masks=mask_tensor)
            mask0 = pipeline.vace_encode_masks(mask_tensor, ref_images=None)
            vace_context_latent = pipeline.vace_latent(vace_context0, mask0)[0]
            
            vace_context_patchified = patchify([vace_context_latent], patch_size=(1,2,2))[0]

            # Move to CPU for saving and convert to float16 to reduce file size
            text_embed_cpu = text_embed.detach().cpu()
            latents_cpu = latents.detach().cpu()
            vace_context_cpu = vace_context_patchified.detach().to(dtype=torch.float16).cpu()

            # Build JSON side-info similar to prepare_energon script
            C, T, H, W = video_tensor.shape[1:]  # 1,C,T,H,W
            json_data = {
                "video_path": video_path,
                "processed_frames": int(T),
                "processed_height": int(H),
                "processed_width": int(W),
                "caption": prompt,
                "deterministic_latents": bool(not args.stochastic),
                "model_version": "wan2.1",
                "resize_settings": {
                    "target_size": target_size,
                    "resize_mode": args.resize_mode,
                    "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                    "center_crop": bool(args.center_crop),
                },
            }
            sample = {
                "__key__": f"{idx:06}",
                "pickle": pickle.dumps(text_embed_cpu),
                "pth": latents_cpu,
                "context.pth": vace_context_cpu,
                "json": json_data,
            }
            sink.write(sample)
            written += 1
    
    print(f"Done writing {written} VACE samples as shards.")

if __name__ == "__main__":
    main()
