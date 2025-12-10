#!/usr/bin/env python3
"""
Batch process all videos from all_mixkit subdirectories using RAM + Grounding DINO + SAM2
Reads frame ranges from video_mixkit.json files in subdirectories
"""
import os
import sys
import json
from pathlib import Path
import subprocess
from tqdm import tqdm

# Import the segmentation pipeline
sys.path.insert(0, str(Path.home() / 'RAM_DINO_SAM'))
from automatic_segmentation import AutomaticSegmentationPipeline, process_video

def find_json_files(root_dir):
    """Find all video_mixkit.json files in subdirectories"""
    json_files = []
    root_path = Path(root_dir).expanduser()
    
    for json_path in root_path.rglob('video_mixkit.json'):
        if json_path.is_file():
            json_files.append(json_path)
    
    return sorted(json_files)

def process_videos_from_json(json_path, input_base_dir, output_base_dir, ram_dino_sam_dir, pipeline):
    """Process all videos listed in a JSON file with their frame ranges"""
    print(f"\n{'='*80}")
    print(f"Processing JSON: {json_path.relative_to(input_base_dir)}")
    print(f"{'='*80}\n")
    
    with open(json_path, 'r') as f:
        video_entries = json.load(f)
    
    # Group entries by video path
    video_groups = {}
    for entry in video_entries:
        video_path = entry['path']
        if video_path not in video_groups:
            video_groups[video_path] = []
        video_groups[video_path].append(entry)
    
    stats = {'successful': 0, 'failed': 0, 'skipped': 0}
    
    # Collect all meta entries for the meta.json
    all_meta_entries = []
    
    # Process each video
    for video_path, entries in video_groups.items():
        full_video_path = input_base_dir / video_path
        
        if not full_video_path.exists():
            print(f"⚠️  Video not found: {full_video_path}")
            stats['failed'] += len(entries)
            continue
        
        # Create output directory maintaining the subdirectory structure
        relative_path = Path(video_path).parent
        video_name = Path(video_path).stem
        output_dir = output_base_dir / relative_path / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Video: {video_path}")
        print(f"Segments: {len(entries)}")
        print(f"Output: {output_dir.relative_to(output_base_dir)}")
        print(f"{'='*80}\n")
        
        # Process each segment
        for idx, entry in enumerate(entries, 1):
            frame_range = entry['frame_idx']
            start_frame, end_frame = frame_range.split(':')
            
            # Create segment-specific output directory
            segment_output_dir = output_dir / f"segment_{start_frame}_{end_frame}"
            
            # Calculate frame index for meta.json (0-based relative to segment)
            segment_length = int(end_frame) - int(start_frame)
            
            # Check if already processed
            if (segment_output_dir / "segmentation_complete.txt").exists():
                print(f"  [{idx}/{len(entries)}] ✓ Skipped (already done): frames {start_frame}-{end_frame}")
                stats['skipped'] += 1
                
                # Still add to meta entries if successful
                original_video_path = segment_output_dir / "original_video.mp4"
                if original_video_path.exists():
                    relative_output_path = original_video_path.relative_to(output_base_dir)
                    meta_entry = {
                        "path": str(relative_output_path),
                        "frame_idx": f"0:{segment_length}",
                        "cap": entry.get('cap', '')
                    }
                    all_meta_entries.append(meta_entry)
                continue
            
            segment_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  [{idx}/{len(entries)}] Processing: frames {start_frame}-{end_frame}")
            
            # Process video segment directly using shared pipeline
            try:
                process_video(
                    video_path=str(full_video_path),
                    output_dir=str(segment_output_dir),
                    pipeline=pipeline,
                    text_prompt=None,
                    use_ram=True,
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    start_time=None,
                    end_time=None,
                    box_threshold=0.25,
                    text_threshold=0.25
                )
                
                # Mark as complete
                with open(segment_output_dir / "segmentation_complete.txt", "w") as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Frames: {start_frame}-{end_frame}\n")
                    f.write(f"Status: Success\n")
                
                print(f"  [{idx}/{len(entries)}] ✅ Success: frames {start_frame}-{end_frame}")
                stats['successful'] += 1
                
                # Add to meta entries
                original_video_path = segment_output_dir / "original_video.mp4"
                if original_video_path.exists():
                    relative_output_path = original_video_path.relative_to(output_base_dir)
                    meta_entry = {
                        "path": str(relative_output_path),
                        "frame_idx": f"0:{segment_length}",
                        "cap": entry.get('cap', '')
                    }
                    all_meta_entries.append(meta_entry)
                
            except Exception as e:
                print(f"  [{idx}/{len(entries)}] ❌ Failed: frames {start_frame}-{end_frame}")
                print(f"      Error: {str(e)}")
                
                # Log error
                with open(segment_output_dir / "segmentation_error.txt", "w") as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Frames: {start_frame}-{end_frame}\n")
                    f.write(f"Status: Failed\n")
                    f.write(f"Error: {str(e)}\n")
                
                stats['failed'] += 1
                continue
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Processing interrupted by user")
                raise
        
        print(f"\n✅ Completed all segments for {video_path}")
    
    # Save meta.json for this JSON file's output
    if all_meta_entries:
        # Determine the output directory for the meta.json
        # Use the parent directory of the first entry to determine where to save
        if all_meta_entries:
            # Save meta.json in the same directory as the JSON file's processed outputs
            json_relative = json_path.relative_to(input_base_dir).parent
            meta_output_dir = output_base_dir / json_relative
            meta_output_path = meta_output_dir / "meta.json"
            
            with open(meta_output_path, 'w') as f:
                json.dump(all_meta_entries, f, indent=2)
            
            print(f"\n📝 Created meta.json with {len(all_meta_entries)} entries: {meta_output_path.relative_to(output_base_dir)}")
    
    return stats


def main():
    # Set up directories
    input_base_dir = Path.home() / 'all_mixkit'
    output_base_dir = Path.home() / 'all_mixkit_segmented'
    ram_dino_sam_dir = Path.home() / 'RAM_DINO_SAM'
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Batch Video Processing with RAM + Grounding DINO + SAM2")
    print(f"{'='*80}")
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}\n")
    
    # Find all JSON files
    print("Searching for video_mixkit.json files...")
    json_files = find_json_files(input_base_dir)
    
    if not json_files:
        print(f"❌ No video_mixkit.json files found in {input_base_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(json_files)} JSON file(s):")
    for json_file in json_files:
        print(f"  - {json_file.relative_to(input_base_dir)}")
    
    # Initialize pipeline once for all processing
    print(f"\n{'='*80}")
    print("Initializing models (RAM + Grounding DINO + SAM2)...")
    print(f"{'='*80}\n")
    
    pipeline = AutomaticSegmentationPipeline(
        sam2_checkpoint=str(ram_dino_sam_dir / 'models/sam2_hiera_large.pt'),
        sam2_config='sam2_hiera_l.yaml',
        grounding_dino_config=str(ram_dino_sam_dir / 'models/GroundingDINO_SwinT_OGC.py'),
        grounding_dino_checkpoint=str(ram_dino_sam_dir / 'models/groundingdino_swint_ogc.pth'),
        ram_checkpoint=str(ram_dino_sam_dir / 'models/ram_plus_swin_large_14m.pth'),
        device='cuda'
    )
    
    print("\n✅ Models loaded successfully! Processing videos...\n")
    
    # Process each JSON file
    total_stats = {'successful': 0, 'failed': 0, 'skipped': 0}
    
    for json_file in json_files:
        try:
            stats = process_videos_from_json(json_file, input_base_dir, output_base_dir, ram_dino_sam_dir, pipeline)
            total_stats['successful'] += stats['successful']
            total_stats['failed'] += stats['failed']
            total_stats['skipped'] += stats['skipped']
        except KeyboardInterrupt:
            print("\n\n⚠️  Processing interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total segments processed:")
    print(f"  ✅ Successful: {total_stats['successful']}")
    print(f"  ❌ Failed: {total_stats['failed']}")
    print(f"  ⏭️  Skipped: {total_stats['skipped']}")
    print(f"  📊 Total: {sum(total_stats.values())}")
    print(f"\nResults saved to: {output_base_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

