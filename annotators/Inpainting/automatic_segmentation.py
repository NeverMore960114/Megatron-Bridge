"""
Automatic Instance Segmentation Pipeline using RAM + Grounding DINO + SAM2

This script performs fully automatic instance segmentation without any manual annotation:
1. RAM (Recognize Anything Model) - Automatically generates image tags
2. Grounding DINO - Detects objects based on generated tags
3. SAM2 - Segments detected objects using bounding boxes as prompts

No human annotation required!
"""
import os
os.environ['HF_HOME'] = '/home/tanya/.huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/tanya/.huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/home/tanya/.huggingface/hub'

import torch
import numpy as np
import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import supervision as sv
from typing import List, Dict, Tuple

# SAM2 imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tempfile
import shutil

# Optional: RAM and Grounding DINO imports (will check if available)
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    print("Warning: Grounding DINO not available. Install with:")
    print("pip install groundingdino-py")
    GROUNDING_DINO_AVAILABLE = False

try:
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram.transform import get_transform as ram_transform
    RAM_AVAILABLE = True
except ImportError:
    print("Warning: RAM not available. Will use manual text prompts.")
    RAM_AVAILABLE = False


class AutomaticSegmentationPipeline:
    """Pipeline for automatic instance segmentation using RAM + Grounding DINO + SAM2"""
    
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str = "sam2_hiera_l.yaml",
        grounding_dino_config: str = None,
        grounding_dino_checkpoint: str = None,
        ram_checkpoint: str = None,
        device: str = "cuda"
    ):
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Load SAM2 for images
        print("Loading SAM2...")
        self.sam2_predictor = SAM2ImagePredictor(
            build_sam2(sam2_config, sam2_checkpoint, device=device)
        )
        
        # Load Grounding DINO
        self.grounding_dino = None
        if GROUNDING_DINO_AVAILABLE and grounding_dino_config and grounding_dino_checkpoint:
            print("Loading Grounding DINO...")
            self.grounding_dino = GroundingDINOModel(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=device
            )
        
        # Load RAM
        self.ram_model = None
        if RAM_AVAILABLE and ram_checkpoint:
            print("Loading RAM...")
            self.ram_model = ram_plus(
                pretrained=ram_checkpoint,
                image_size=384,
                vit='swin_l'
            )
            self.ram_model.eval()
            self.ram_model = self.ram_model.to(device)
    
    def generate_tags_with_ram(self, image: np.ndarray) -> List[str]:
        """Generate image tags using RAM model"""
        if self.ram_model is None:
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Preprocess image for RAM using the transform
        transform = ram_transform(image_size=384)
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Generate tags using the model
        with torch.no_grad():
            tags, tags_chinese = self.ram_model.generate_tag(image_tensor)
        
        # Parse tags - they come as a string separated by |
        tag_list = [tag.strip() for tag in tags[0].split('|') if tag.strip()]
        
        return tag_list
    
    def detect_objects_with_grounding_dino(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        min_area_ratio: float = 0.20,
        max_area_ratio: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Detect objects using Grounding DINO
        
        Args:
            min_area_ratio: Minimum box area as ratio of image area (default: 0.20)
            max_area_ratio: Maximum box area as ratio of image area (default: 0.50)
        
        Returns:
            boxes: (N, 4) array of bounding boxes in xyxy format
            scores: (N,) array of confidence scores
            labels: List of N labels
        """
        if self.grounding_dino is None:
            return np.array([]), np.array([]), []
        
        # Detect objects
        detections = self.grounding_dino.predict_with_classes(
            image=image,
            classes=[text_prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Extract results
        boxes = detections.xyxy if len(detections) > 0 else np.array([])
        scores = detections.confidence if len(detections) > 0 else np.array([])
        labels = detections.class_id if len(detections) > 0 else []
        
        # Filter boxes by area
        if len(boxes) > 0:
            image_area = image.shape[0] * image.shape[1]
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area_ratios = box_areas / image_area
            
            # Keep boxes within area ratio range
            valid_mask = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            labels = [label for i, label in enumerate(labels) if valid_mask[i]]
            
            print(f"Filtered boxes: {valid_mask.sum()}/{len(valid_mask)} boxes kept (area between {min_area_ratio*100}% and {max_area_ratio*100}%)")
            
            # Keep only the box with highest score
            if len(boxes) > 0:
                best_idx = np.argmax(scores)
                boxes = boxes[best_idx:best_idx+1]
                scores = scores[best_idx:best_idx+1]
                labels = [labels[best_idx]]
                print(f"Selected box with highest score: {scores[0]:.3f}")
        
        return boxes, scores, labels
    
    def segment_with_sam2(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Segment objects using SAM2 with bounding box prompts
        
        Args:
            image: Input image (H, W, 3)
            boxes: Bounding boxes in xyxy format (N, 4)
        
        Returns:
            List of binary masks, one for each box
        """
        if len(boxes) == 0:
            return []
        
        # Set image
        self.sam2_predictor.set_image(image)
        
        masks = []
        for box in boxes:
            # SAM expects box in xyxy format
            mask, score, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # Add batch dimension
                multimask_output=False,
            )
            masks.append(mask[0])  # Take first (and only) mask
        
        return masks
    
    def process_image(
        self,
        image: np.ndarray,
        text_prompt: str = None,
        use_ram: bool = True,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        min_area_ratio: float = 0.20,
        max_area_ratio: float = 0.50
    ) -> Dict:
        """
        Process a single image through the full pipeline
        
        Args:
            image: Input image (H, W, 3) in BGR format
            text_prompt: Optional text prompt. If None and use_ram=True, will generate automatically
            use_ram: Whether to use RAM for automatic tag generation
            box_threshold: Grounding DINO box threshold
            text_threshold: Grounding DINO text threshold
        
        Returns:
            Dictionary containing:
                - tags: Generated or provided tags
                - boxes: Detected bounding boxes
                - scores: Detection confidence scores
                - masks: Instance segmentation masks
                - labels: Object labels
        """
        # Step 1: Generate tags with RAM (if enabled and no prompt provided)
        tags = []
        if text_prompt is None and use_ram:
            tags = self.generate_tags_with_ram(image)
            text_prompt = " . ".join(tags) if tags else "object"
            print(f"Generated tags: {tags}")
        elif text_prompt is None:
            text_prompt = "object"
        
        # Step 2: Detect objects with Grounding DINO
        boxes, scores, labels = self.detect_objects_with_grounding_dino(
            image,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio
        )
        
        print(f"Detected {len(boxes)} objects")
        
        # Step 3: Segment with SAM2
        masks = self.segment_with_sam2(image, boxes)
        
        return {
            'tags': tags,
            'text_prompt': text_prompt,
            'boxes': boxes,
            'scores': scores,
            'masks': masks,
            'labels': labels
        }


def visualize_results(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: List[np.ndarray],
    scores: np.ndarray,
    labels: List[str] = None
) -> np.ndarray:
    """Visualize detection and segmentation results"""
    vis_image = image.copy()
    
    # Generate random colors for each instance
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    # Draw masks
    for idx, mask in enumerate(masks):
        color = colors[idx].tolist()
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        # Overlay with transparency
        vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)
    
    # Draw bounding boxes
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[idx].tolist()
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label_text = f"{labels[idx] if labels else 'obj'}: {score:.2f}"
        cv2.putText(vis_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image


def extract_video_frames(video_path: str, output_dir: str, start_frame: int, end_frame: int) -> Tuple[float, int, int]:
    """Extract frames from video to directory"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame == -1:
        end_frame = total_frames
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    extracted_count = 0
    
    print(f"Extracting frames {start_frame} to {end_frame}...")
    pbar = tqdm(total=end_frame - start_frame, desc="Extracting frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break
        
        if frame_idx >= start_frame:
            # Save frame with relative index (starting from 0)
            frame_filename = os.path.join(output_dir, f"{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            pbar.update(1)
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    return fps, width, height


def process_video(
    video_path: str,
    output_dir: str,
    pipeline: AutomaticSegmentationPipeline,
    text_prompt: str = None,
    use_ram: bool = True,
    start_frame: int = 0,
    end_frame: int = -1,
    start_time: float = None,
    end_time: float = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    min_area_ratio: float = 0.20,
    max_area_ratio: float = 0.50
):
    """
    Process video with automatic segmentation and propagation.
    Uses RAM + Grounding DINO on first frame, then SAM2 propagates through remaining frames.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory
        pipeline: AutomaticSegmentationPipeline instance
        text_prompt: Manual text prompt (if not using RAM)
        use_ram: Whether to use RAM for tag generation
        start_frame: Starting frame index (overridden by start_time if provided)
        end_frame: Ending frame index (overridden by end_time if provided)
        start_time: Starting timestamp in seconds
        end_time: Ending timestamp in seconds
        box_threshold: Grounding DINO box threshold
        text_threshold: Grounding DINO text threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert timestamps to frame indices if provided
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    if start_time is not None:
        start_frame = int(start_time * fps)
        print(f"Start time {start_time}s -> frame {start_frame}")
    
    if end_time is not None:
        end_frame = int(end_time * fps)
        print(f"End time {end_time}s -> frame {end_frame}")
    
    if end_frame == -1:
        end_frame = total_frames
    
    # Create temporary directory for extracted frames
    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    frames_dir = os.path.join(temp_dir, "frames")
    
    try:
        # Extract frames
        fps, width, height = extract_video_frames(video_path, frames_dir, start_frame, end_frame)
        
        # Read first frame for detection
        first_frame_path = os.path.join(frames_dir, "00000.jpg")
        first_frame = cv2.imread(first_frame_path)
        
        print("\n=== Step 1: Detecting objects in first frame ===")
        
        # Step 1: Generate tags with RAM (if enabled)
        tags = []
        if text_prompt is None and use_ram:
            tags = pipeline.generate_tags_with_ram(first_frame)
            text_prompt = " . ".join(tags) if tags else "object"
            print(f"Generated tags: {tags}")
        elif text_prompt is None:
            text_prompt = "object"
        
        # Step 2: Detect objects with Grounding DINO
        boxes, scores, labels = pipeline.detect_objects_with_grounding_dino(
            first_frame,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio
        )
        
        print(f"Detected {len(boxes)} objects")
        
        if len(boxes) == 0:
            print("Warning: No objects detected! Try lowering thresholds or providing specific prompts.")
            return
        
        # Print detection summary
        for i, (box, score) in enumerate(zip(boxes, scores)):
            print(f"  Object {i+1}: confidence={score:.3f}, box={box.astype(int)}")
        
        print("\n=== Step 2: Initializing SAM2 video propagation ===")
        
        # Step 3: Initialize SAM2 video predictor
        video_predictor = build_sam2_video_predictor(
            pipeline.sam2_config,
            pipeline.sam2_checkpoint,
            device=pipeline.device
        )
        
        inference_state = video_predictor.init_state(video_path=frames_dir)
        
        # Add all detected objects to the first frame
        for obj_id, box in enumerate(boxes, start=1):
            # Convert box to center point + box format for SAM2
            x1, y1, x2, y2 = box
            
            # Add box prompt to SAM2
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,  # First frame
                obj_id=obj_id,
                box=box,
            )
        
        print(f"Added {len(boxes)} objects to track")
        print("\n=== Step 3: Propagating masks through video ===")
        
        # Step 4: Propagate through video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        print("\n=== Step 4: Saving results ===\n")
        
        # Step 5: Create output videos and save masks
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, "original_video.mp4")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Setup mask video writers for each object
        mask_video_writers = {}
        src_video_writers = {}
        for obj_id in range(1, len(boxes) + 1):
            mask_video_path = os.path.join(output_dir, f"mask_obj_{obj_id}.mp4")
            mask_video_writers[obj_id] = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height), isColor=False)
            
            # Setup source video writer with inverse mask applied (for inpainting)
            src_video_path = os.path.join(output_dir, f"src_video_obj_{obj_id}.mp4")
            src_video_writers[obj_id] = cv2.VideoWriter(src_video_path, fourcc, fps, (width, height))
        
        # Generate random colors for each object
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
        
        num_frames = end_frame - start_frame
        for frame_idx in tqdm(range(num_frames), desc="Saving results"):
            # Read frame
            frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            frame = cv2.imread(frame_path)
            
            # Write original frame to video
            video_writer.write(frame)
            
            # Process masks if available
            if frame_idx in video_segments:
                for obj_id in sorted(video_segments[frame_idx].keys()):
                    mask = video_segments[frame_idx][obj_id][0] # Get mask
                    
                    # Write mask frame to video
                    mask_img = (mask * 255).astype(np.uint8)
                    mask_video_writers[obj_id].write(mask_img)
                    
                    # Create source video with inverse mask applied (zeroing out the object for inpainting)
                    bool_mask = mask > 0
                    src_frame = frame.copy()
                    src_frame[bool_mask] = 128  # Gray out the masked region
                    src_video_writers[obj_id].write(src_frame)
        
        video_writer.release()
        for obj_id, mask_writer in mask_video_writers.items():
            mask_writer.release()
        for obj_id, src_writer in src_video_writers.items():
            src_writer.release()
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"{'='*60}")
        print(f"Video segment: frames {start_frame} to {end_frame}")
        if start_time is not None or end_time is not None:
            print(f"Time segment: {start_time if start_time else 0}s to {end_time if end_time else end_frame/fps}s")
        print(f"Detected and tracked {len(boxes)} objects")
        print(f"\nOutputs:")
        print(f"  Original video: {output_video_path}")
        for obj_id in range(1, len(boxes) + 1):
            mask_video_path = os.path.join(output_dir, f"mask_obj_{obj_id}.mp4")
            src_video_path = os.path.join(output_dir, f"src_video_obj_{obj_id}.mp4")
            print(f"  Mask video (obj {obj_id}): {mask_video_path}")
            print(f"  Source video with inverse mask (obj {obj_id}): {src_video_path}")
        
        # Save detection info
        info_path = os.path.join(output_dir, "detection_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Frames: {start_frame} to {end_frame}\n")
            if start_time is not None or end_time is not None:
                f.write(f"Time: {start_time if start_time else 0}s to {end_time if end_time else end_frame/fps}s\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"\nGenerated tags: {', '.join(tags) if tags else 'N/A'}\n")
            f.write(f"Text prompt used: {text_prompt}\n")
            f.write(f"\nDetected {len(boxes)} objects:\n")
            for i, (box, score) in enumerate(zip(boxes, scores)):
                f.write(f"  Object {i+1}: confidence={score:.3f}, box={box.astype(int).tolist()}\n")
        print(f"  Detection info: {info_path}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary files")

def process_image_single(
    image_path: str,
    output_dir: str,
    pipeline: AutomaticSegmentationPipeline,
    text_prompt: str = None,
    use_ram: bool = True,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    min_area_ratio: float = 0.20,
    max_area_ratio: float = 0.50
):  
    """Process a single image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    # Process
    results = pipeline.process_image(
        image,
        text_prompt=text_prompt,
        use_ram=use_ram,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )  
    
    # Visualize
    vis_image = visualize_results(
        image,
        results['boxes'],
        results['masks'],
        results['scores'],
        results['labels']
    )
    
    # Save results
    output_path = os.path.join(output_dir, "segmentation_result.jpg")
    cv2.imwrite(output_path, vis_image)
    
    # Save individual masks
    for mask_idx, mask in enumerate(results['masks']):
        mask_img = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"mask_{mask_idx}.png")
        cv2.imwrite(mask_path, mask_img)
    
    print(f"\nProcessing complete!")
    print(f"Detected objects: {len(results['boxes'])}")
    print(f"Tags used: {results['text_prompt']}")
    print(f"Output: {output_path}")
    print(f"Masks saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic instance segmentation using RAM + Grounding DINO + SAM2"
    )
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or video")
    parser.add_argument("--output-dir", type=str, default="auto_segmentation_output",
                       help="Output directory")
    parser.add_argument("--mode", type=str, choices=["image", "video"], default="image",
                       help="Processing mode")
    
    # SAM2 arguments
    parser.add_argument("--sam2-checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_l.yaml",
                       help="SAM2 config file")
    
    # Grounding DINO arguments
    parser.add_argument("--grounding-dino-config", type=str,
                       help="Path to Grounding DINO config file")
    parser.add_argument("--grounding-dino-checkpoint", type=str,
                       help="Path to Grounding DINO checkpoint")
    
    # RAM arguments
    parser.add_argument("--ram-checkpoint", type=str,
                       help="Path to RAM checkpoint")
    parser.add_argument("--no-ram", action="store_true", default=False,
                       help="Disable RAM and use manual text prompt")
    # Detection parameters
    parser.add_argument("--text-prompt", type=str, default=None,
                       help="Text prompt for detection (if not using RAM)")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                       help="Grounding DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                       help="Grounding DINO text threshold")
    parser.add_argument("--min-area-ratio", type=float, default=0.20,
                       help="Minimum box area as ratio of image area (default: 0.20)")
    parser.add_argument("--max-area-ratio", type=float, default=0.50,
                       help="Maximum box area as ratio of image area (default: 0.50)")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                       help="Grounding DINO text threshold")
    
    # Video-specific arguments
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Starting frame for video processing (overridden by --start-time)")
    parser.add_argument("--end-frame", type=int, default=-1,
                       help="Ending frame for video processing (-1 for end, overridden by --end-time)")
    parser.add_argument("--start-time", type=float, default=None,
                       help="Starting timestamp in seconds (overrides --start-frame)")
    parser.add_argument("--end-time", type=float, default=None,
                       help="Ending timestamp in seconds (overrides --end-frame)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device to run on")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutomaticSegmentationPipeline(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint)
    # Process based on mode
    if args.mode == "image":
        process_image_single(
            args.input,
            args.output_dir,
            pipeline,
            text_prompt=args.text_prompt,
            use_ram=not args.no_ram,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio
        )
    else:  # video
        process_video(
            args.input,
            args.output_dir,
            pipeline,
            text_prompt=args.text_prompt,
            use_ram=not args.no_ram,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio
        )


if __name__ == "__main__":
    main()
