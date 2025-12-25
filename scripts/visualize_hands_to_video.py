#!/usr/bin/env python3
"""
Visualize hand 2D keypoints from HaWoR reconstruction and encode to MP4 video.
Left hand (track 0): darker red, Right hand (track 1): darker green.
Uses ffmpeg stdin pipe to avoid temporary image files.
FPS matches the extraction rate (default 30fps as used in detect_track_video.py).
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from glob import glob

import numpy as np
import cv2
import torch
from natsort import natsorted
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.models.mano_wrapper import MANO


def load_cam_space_chunk(json_path):
    """Load MANO parameters from cam_space JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    R_root = np.array(data["init_root_orient"])
    R_hand = np.array(data["init_hand_pose"])
    t_root = np.array(data["init_trans"])
    betas = np.array(data["init_betas"])

    # Normalize shapes
    R_root = np.squeeze(R_root)
    R_hand = np.squeeze(R_hand)
    t_root = np.squeeze(t_root)
    betas = np.squeeze(betas)

    if R_root.ndim == 2 and R_root.shape == (3, 3):
        R_root = R_root[None, ...]
    if R_hand.ndim == 3 and R_hand.shape == (15, 3, 3):
        R_hand = R_hand[None, ...]
    if t_root.ndim == 1 and t_root.shape == (3,):
        t_root = t_root[None, ...]
    if betas.ndim == 1:
        betas = betas[None, ...]

    T = R_root.shape[0]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, T, axis=0)

    return R_root, R_hand, t_root, betas


def create_mano_model(is_left=False, device="cpu"):
    """Create and return a MANO model instance."""
    if is_left:
        mano_cfg = {
            'data_dir': '_DATA/data_left/',
            'model_path': '_DATA/data_left/mano_left',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'num_betas': 10,
            'create_body_pose': False,
            'is_rhand': False,
        }
    else:
        mano_cfg = {
            'data_dir': '_DATA/data/',
            'model_path': '_DATA/data/mano',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'num_betas': 10,
            'create_body_pose': False,
        }

    mano = MANO(**mano_cfg).to(device)

    # Fix MANO left shapedirs bug
    if is_left and hasattr(mano, 'shapedirs'):
        with torch.no_grad():
            mano.shapedirs[:, 0, :] *= -1

    return mano


def mano_forward(mano_model, R_root, R_hand, t_root, betas, device="cpu"):
    """Run MANO forward pass with pre-created model."""
    # Convert to torch tensors
    global_orient = torch.from_numpy(R_root).float().to(device).unsqueeze(1)
    hand_pose = torch.from_numpy(R_hand).float().to(device)
    transl = torch.from_numpy(t_root).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)

    with torch.no_grad():
        out = mano_model(global_orient=global_orient, hand_pose=hand_pose, 
                        betas=betas_t, transl=transl, pose2rot=False)

    joints = out.joints.detach().cpu().numpy()
    return joints


def project_points_cam(points, fx, fy, cx, cy):
    """Project 3D points to 2D image coordinates."""
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    eps = 1e-6
    Zc = np.clip(Z, eps, None)
    u = fx * (X / Zc) + cx
    v = fy * (Y / Zc) + cy
    return np.stack([u, v], axis=-1), Z


def draw_hand_skeleton(img, keypoints_2d, color=(0, 150, 0), thickness=2):
    """
    Draw hand skeleton on image.
    OpenPose hand 21 keypoints topology.
    """
    # Hand bone connections
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
    ]

    # Draw skeleton lines
    for i, j in edges:
        pt1 = keypoints_2d[i]
        pt2 = keypoints_2d[j]
        if np.isfinite(pt1).all() and np.isfinite(pt2).all():
            cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                    color, thickness, cv2.LINE_AA)

    # Draw keypoints
    for pt in keypoints_2d:
        if np.isfinite(pt).all():
            cv2.circle(img, tuple(pt.astype(int)), 3, color, -1, cv2.LINE_AA)

    return img


def load_captions(captions_json_path, segment_def_json_path, task_name):
    """
    Load captions for a specific task from the captions JSON file.
    
    Args:
        captions_json_path: Path to clip_1.json (captions file)
        segment_def_json_path: Path to clip_1_segment_def.json (segment definitions)
        task_name: Task directory name (e.g., '00_task')
        
    Returns:
        List of clips with 'start', 'end', 'desc' for this task, or [] if not found
    """
    try:
        with open(captions_json_path, 'r') as f:
            captions_data = json.load(f)
        
        # Find the video entry matching this task
        video_key = f"{task_name}.mp4"
        for video_data in captions_data:
            if video_data.get("video") == video_key:
                return video_data.get("clips", [])
        
        print(f"Warning: No captions found for {video_key} in {captions_json_path}")
        return []
        
    except Exception as e:
        print(f"Warning: Failed to load captions from {captions_json_path}: {e}")
        return []


def get_caption_for_frame(frame_idx, clips, extraction_fps=30, captions_fps=50):
    """
    Find the caption for a given frame index.
    
    Args:
        frame_idx: Local frame index in the extracted images (at extraction_fps)
        clips: List of clip dicts with 'start', 'end', 'desc' (frame ranges at captions_fps)
        extraction_fps: FPS of the extracted images (default 30)
        captions_fps: FPS of the original video that captions refer to (default 50)
        
    Returns:
        Caption text string, or None if frame is outside all clip ranges
    """
    # Convert frame_idx from extraction_fps to captions_fps
    # frame_in_original = frame_in_extracted * (captions_fps / extraction_fps)
    frame_idx_converted = frame_idx * (captions_fps / extraction_fps)
    
    for clip in clips:
        if clip["start"] <= frame_idx_converted <= clip["end"]:
            return clip["desc"]
    return None


def wrap_text(text, font, font_scale, thickness, max_width):
    """
    Wrap text into multiple lines to fit within max_width.
    
    Args:
        text: Text to wrap
        font: OpenCV font constant
        font_scale: Font scale
        thickness: Font thickness
        max_width: Maximum width in pixels
        
    Returns:
        List of text lines
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
        
        if text_size <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def draw_caption(img, text, font_scale=0.7, padding=10):
    """
    Draw caption text at the bottom of the image with semi-transparent background.
    
    Args:
        img: Image to draw on (modified in-place)
        text: Caption text to draw
        font_scale: Font scale (default 0.7)
        padding: Padding around text in pixels (default 10)
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)      # Black background
    
    # Wrap text to fit within frame width
    max_width = w - 2 * padding
    lines = wrap_text(text, font, font_scale, thickness, max_width)
    
    # Calculate text box dimensions
    line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1]
    line_spacing = 5
    total_height = len(lines) * (line_height + line_spacing) + 2 * padding
    
    # Draw semi-transparent background at bottom
    y_start = h - total_height
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_start), (w, h), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw text lines
    y = y_start + padding + line_height
    for line in lines:
        cv2.putText(img, line, (padding, y), font, font_scale, 
                   color, thickness, cv2.LINE_AA)
        y += line_height + line_spacing


def load_track_data(cam_space_dir, track_id, is_left, focal, cx, cy, device="cpu"):
    """
    Load all cam_space data for a track and project to 2D.
    Returns: dict mapping frame_idx -> keypoints_2d (21, 2)
    """
    track_dir = cam_space_dir / str(track_id)
    if not track_dir.exists():
        return {}

    # Create MANO model once for this track
    mano_model = create_mano_model(is_left=is_left, device=device)

    frame_data = {}
    chunk_files = natsorted(glob(str(track_dir / '*.json')))

    for json_file in chunk_files:
        try:
            # Load MANO parameters
            R_root, R_hand, t_root, betas = load_cam_space_chunk(json_file)

            # Generate 3D joints (reuse same model)
            joints_3d = mano_forward(mano_model, R_root, R_hand, t_root, betas, device=device)

            # Get frame range from filename
            stem = Path(json_file).stem
            s_idx, e_idx = map(int, stem.split('_'))

            # Project to 2D for each frame
            for i in range(len(joints_3d)):
                frame_idx = s_idx + i
                joints_2d, depth = project_points_cam(joints_3d[i], focal, focal, cx, cy)
                joints_2d[depth <= 1e-6] = np.nan
                frame_data[frame_idx] = joints_2d

        except Exception as e:
            print(f"Warning: Failed to process {json_file}: {e}")
            continue

    return frame_data


def visualize_to_video(task_dir, output_path, fps=30, device="cpu", 
                       captions_json=None, segment_def_json=None, 
                       caption_font_scale=0.7, caption_padding=10,
                       captions_fps=50):
    """
    Visualize hand keypoints and encode to video via ffmpeg pipe.
    
    Args:
        task_dir: Path to task directory (e.g., example/clip_1/00_task)
        output_path: Output video path (e.g., output.mp4)
        fps: Video frame rate (default 30, matching detect_track_video.py extraction)
        device: torch device for MANO
        captions_json: Path to captions JSON file (optional)
        segment_def_json: Path to segment definition JSON file (optional)
        caption_font_scale: Font scale for captions (default 0.7)
        caption_padding: Padding around caption text (default 10)
        captions_fps: FPS of the original video that captions refer to (default 50)
    """
    task_path = Path(task_dir)
    
    # Load focal length
    focal_path = task_path / 'est_focal.txt'
    if focal_path.exists():
        focal = float(focal_path.read_text().strip())
    else:
        print(f"Warning: {focal_path} not found, using default focal=600")
        focal = 600.0

    # Load captions if provided
    clips = []
    if captions_json and segment_def_json:
        task_name = task_path.name  # e.g., '00_task'
        clips = load_captions(captions_json, segment_def_json, task_name)
        if clips:
            print(f"Loaded {len(clips)} caption clips for {task_name}")

    # Load images
    img_dir = task_path / 'extracted_images'
    img_files = natsorted(glob(str(img_dir / '*.jpg'))) or natsorted(glob(str(img_dir / '*.png')))
    
    if not img_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Get image dimensions
    first_img = cv2.imread(img_files[0])
    h, w = first_img.shape[:2]
    cx, cy = w / 2, h / 2

    print(f"Processing {len(img_files)} frames at {w}x{h}, focal={focal:.1f}, fps={fps}")

    # Load cam_space data for both hands
    # Track 0 = left hand, Track 1 = right hand (hardcoded)
    cam_space_dir = task_path / 'cam_space'
    
    print("Loading left hand (track 0)...")
    left_data = load_track_data(cam_space_dir, 0, is_left=True, 
                                focal=focal, cx=cx, cy=cy, device=device)
    
    print("Loading right hand (track 1)...")
    right_data = load_track_data(cam_space_dir, 1, is_left=False,
                                 focal=focal, cx=cx, cy=cy, device=device)

    print(f"Left hand: {len(left_data)} frames, Right hand: {len(right_data)} frames")

    # Use softer colors: darker red for left, darker green for right (BGR format)
    color_left = (0, 0, 180)    # BGR: darker red
    color_right = (0, 150, 0)   # BGR: darker green

    # FFmpeg command for H.264 encoding
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # Quality (lower = better)
        '-preset', 'medium',
        str(output_path)
    ]

    # Start ffmpeg process
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                           stderr=subprocess.PIPE)

    try:
        # Process frames with progress bar
        for frame_idx in tqdm(range(len(img_files)), desc="Encoding video"):
            img = cv2.imread(img_files[frame_idx])
            if img is None:
                continue

            # Draw left hand (darker red)
            if frame_idx in left_data:
                draw_hand_skeleton(img, left_data[frame_idx], color_left, thickness=2)

            # Draw right hand (darker green)
            if frame_idx in right_data:
                draw_hand_skeleton(img, right_data[frame_idx], color_right, thickness=2)

            # Draw caption if available
            if clips:
                caption = get_caption_for_frame(frame_idx, clips, 
                                               extraction_fps=fps, 
                                               captions_fps=captions_fps)
                if caption:
                    draw_caption(img, caption, font_scale=caption_font_scale, 
                               padding=caption_padding)

            # Write frame to ffmpeg stdin
            proc.stdin.write(img.tobytes())

        # Close stdin to signal end of input
        proc.stdin.close()

        # Wait for ffmpeg to finish
        proc.wait()

        if proc.returncode == 0:
            print(f"✓ Video saved: {output_path}")
        else:
            stderr = proc.stderr.read().decode()
            print(f"✗ FFmpeg error:\n{stderr}")

    except Exception as e:
        proc.kill()
        raise e
    finally:
        if proc.stdin:
            proc.stdin.close()
        if proc.stderr:
            proc.stderr.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HaWoR hand reconstruction results as video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python %(prog)s --dir example/clip_1/00_task --output vis.mp4
  python %(prog)s --dir example/video_0 --output output.mp4 --fps 24
  
Note: Default FPS is 30 to match the frame extraction rate in detect_track_video.py
      Track 0 = left hand (darker red), Track 1 = right hand (darker green)
        """)
    parser.add_argument('--dir', required=True, 
                       help='Task directory containing cam_space/ and extracted_images/')
    parser.add_argument('--output', required=True,
                       help='Output video path (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video frame rate (default: 30, matching extraction)')
    parser.add_argument('--device', default='cpu',
                       help='Device for MANO (cpu or cuda)')
    parser.add_argument('--captions-json', type=str, default=None,
                       help='Path to captions JSON file (e.g., example/clip_1/clip_1.json)')
    parser.add_argument('--segment-def-json', type=str, default=None,
                       help='Path to segment definition JSON file (e.g., example/clip_1/clip_1_segment_def.json)')
    parser.add_argument('--caption-font-scale', type=float, default=0.7,
                       help='Font scale for caption text (default: 0.7)')
    parser.add_argument('--caption-padding', type=int, default=10,
                       help='Padding around caption text in pixels (default: 10)')
    parser.add_argument('--captions-fps', type=float, default=50.0,
                       help='FPS of the original video that captions refer to (default: 50)')
    
    args = parser.parse_args()

    # Validate paths
    task_path = Path(args.dir)
    if not task_path.exists():
        print(f"Error: Directory not found: {task_path}")
        sys.exit(1)

    cam_space = task_path / 'cam_space'
    img_dir = task_path / 'extracted_images'
    
    if not cam_space.exists():
        print(f"Error: cam_space/ not found in {task_path}")
        sys.exit(1)
    
    if not img_dir.exists():
        print(f"Error: extracted_images/ not found in {task_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run visualization
    visualize_to_video(args.dir, args.output, fps=args.fps, device=args.device,
                      captions_json=args.captions_json,
                      segment_def_json=args.segment_def_json,
                      caption_font_scale=args.caption_font_scale,
                      caption_padding=args.caption_padding,
                      captions_fps=args.captions_fps)


if __name__ == '__main__':
    main()
