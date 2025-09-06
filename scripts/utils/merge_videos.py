#!/usr/bin/env python3
"""
Video Frame Averaging Script

This script takes two video files as input and creates a new video where each frame
is the average of the corresponding frames from the two input videos.

Usage:
    python merge_videos.py video1.mp4 video2.mp4 output.mp4
    
Requirements:
    - opencv-python
    - numpy
"""

import cv2
import numpy as np
import argparse
import sys
import os


def merge_videos_with_averaging(video1_path, video2_path, output_path, codec='H264'):
    """
    Merge two videos by averaging corresponding frames.
    
    Args:
        video1_path (str): Path to the first video file
        video2_path (str): Path to the second video file
        output_path (str): Path for the output merged video
        codec (str): Video codec to use ('H264', 'XVID', 'MJPG', 'mp4v')
    """
    # Open video captures
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        raise ValueError(f"Could not open video file: {video1_path}")
    if not cap2.isOpened():
        raise ValueError(f"Could not open video file: {video2_path}")
    
    # Get video properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video 1: {width1}x{height1}, {fps1} FPS, {frame_count1} frames")
    print(f"Video 2: {width2}x{height2}, {fps2} FPS, {frame_count2} frames")
    
    # Check if videos have compatible properties
    if abs(fps1 - fps2) > 0.1:
        print(f"Warning: Videos have different frame rates ({fps1} vs {fps2})")
    
    # Use the smaller dimensions and frame count for compatibility
    output_width = min(width1, width2)
    output_height = min(height1, height2)
    output_fps = min(fps1, fps2)
    max_frames = min(frame_count1, frame_count2)
    
    print(f"Output: {output_width}x{output_height}, {output_fps} FPS, {max_frames} frames")
    
    # Define the codec and create VideoWriter object
    # Try the specified codec first, with fallbacks for compatibility
    codecs_to_try = [codec, 'XVID', 'MJPG', 'mp4v']
    out = None
    
    # Determine file extension and adjust codec selection
    file_ext = os.path.splitext(output_path)[1].lower()
    
    for codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter.fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                break
            else:
                out.release()
        except Exception as e:
            print(f"Failed to use codec {codec_name}: {e}")
            continue
    
    if out is None or not out.isOpened():
        # Try creating as AVI with XVID as last resort
        if file_ext != '.avi':
            avi_path = os.path.splitext(output_path)[0] + '.avi'
            print(f"Trying to create as AVI format: {avi_path}")
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
            out = cv2.VideoWriter(avi_path, fourcc, output_fps, (output_width, output_height))
            if out.isOpened():
                print(f"Successfully created as AVI with XVID codec: {avi_path}")
                output_path = avi_path  # Update the path for the rest of the function
            else:
                out.release()
        
        if out is None or not out.isOpened():
            raise ValueError(f"Could not create output video file with any supported codec: {output_path}")
    
    frame_idx = 0
    
    try:
        while frame_idx < max_frames:
            # Read frames from both videos
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print(f"Reached end of video(s) at frame {frame_idx}")
                break
            
            # Resize frames to match output dimensions if necessary
            if frame1.shape[:2] != (output_height, output_width):
                frame1 = cv2.resize(frame1, (output_width, output_height))
            if frame2.shape[:2] != (output_height, output_width):
                frame2 = cv2.resize(frame2, (output_width, output_height))
            
            # Convert to float for averaging to prevent overflow
            frame1_float = frame1.astype(np.float32)
            frame2_float = frame2.astype(np.float32)
            
            # Average the frames
            averaged_frame = (frame1_float + frame2_float) / 2.0
            
            # Convert back to uint8
            averaged_frame = averaged_frame.astype(np.uint8)
            
            # Write the averaged frame
            out.write(averaged_frame)
            
            frame_idx += 1
            
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                progress = (frame_idx / max_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{max_frames} frames)")
    
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        raise
    
    finally:
        # Release everything
        cap1.release()
        cap2.release()
        out.release()
    
    print(f"Successfully created merged video: {output_path}")
    print(f"Processed {frame_idx} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two videos by averaging corresponding frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python merge_videos.py video1.mp4 video2.mp4 merged_output.mp4
    python merge_videos.py /path/to/video1.avi /path/to/video2.avi output.mp4
    python merge_videos.py video1.mp4 video2.mp4 output.avi --codec XVID

Codec Notes:
    - H264: Modern codec, best quality but may not be available on all systems
    - XVID: Good compatibility, works well with AVI format
    - MJPG: Motion JPEG, larger files but universal compatibility
    - mp4v: MPEG-4, basic compatibility
    
    If the specified codec fails, the script will automatically try fallback codecs.
    For maximum compatibility, use .avi extension with XVID codec.
        """
    )
    
    parser.add_argument("video1", help="Path to the first video file")
    parser.add_argument("video2", help="Path to the second video file")
    parser.add_argument("output", help="Path for the output merged video file")
    parser.add_argument("--codec", "-c", default="H264", 
                       choices=["H264", "XVID", "MJPG", "mp4v"],
                       help="Video codec to use (default: H264)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.video1):
        print(f"Error: Video file not found: {args.video1}")
        sys.exit(1)
    
    if not os.path.exists(args.video2):
        print(f"Error: Video file not found: {args.video2}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        merge_videos_with_averaging(args.video1, args.video2, args.output, args.codec)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
