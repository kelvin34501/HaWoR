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
import threading
from pathlib import Path
from glob import glob

import numpy as np
import cv2
import torch
from natsort import natsorted
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.models.mano_wrapper import MANO


def drain_stderr_pipe(pipe, output_list):
    """Drain stderr pipe in background thread to prevent buffer deadlock."""
    try:
        for line in iter(pipe.readline, b''):
            output_list.append(line.decode('utf-8', errors='replace'))
    except Exception:
        pass
    finally:
        pipe.close()


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


def draw_hand_skeleton(img, keypoints_2d, color=(0, 150, 0), thickness=4):
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
            cv2.circle(img, tuple(pt.astype(int)), 6, color, -1, cv2.LINE_AA)

    return img


def load_captions(captions_json_path, segment_def_json_path, task_name):
    """
    Load captions for a specific task from the captions JSON file.
    
    Args:
        captions_json_path: Path to clip_1.json (captions file)
        segment_def_json_path: Path to clip_1_segment_def.json (segment definitions)
        task_name: Task directory name (e.g., '00_task')
        
    Returns:
        List of clips with 'start', 'end', 'caption', 'description' for this task, or [] if not found
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
        clips: List of clip dicts with 'start', 'end', 'caption', 'description' (frame ranges at captions_fps)
        extraction_fps: FPS of the extracted images (default 30)
        captions_fps: FPS of the original video that captions refer to (default 50)
        
    Returns:
        Dict with 'caption', 'desc', 'start', 'end', 'extracted_start', 'extracted_end' keys, 
        or None if frame is outside all clip ranges
    """
    # Convert frame_idx from extraction_fps to captions_fps
    # frame_in_original = frame_in_extracted * (captions_fps / extraction_fps)
    frame_idx_converted = frame_idx * (captions_fps / extraction_fps)
    
    for clip in clips:
        if clip["start"] <= frame_idx_converted < clip["end"]:
            # Calculate extracted frame range
            extracted_start = int(clip["start"] * extraction_fps / captions_fps)
            extracted_end = int(clip["end"] * extraction_fps / captions_fps)
            return {
                "caption": clip.get("caption", "Unknown"),
                "desc": clip.get("description", ""),
                "start": clip["start"],
                "end": clip["end"],
                "extracted_start": extracted_start,
                "extracted_end": extracted_end
            }
    return None


def wrap_text_pil(text, font, max_width, draw):
    """
    Wrap text into multiple lines to fit within max_width using PIL.
    
    Args:
        text: Text to wrap
        font: PIL ImageFont object
        max_width: Maximum width in pixels
        draw: PIL ImageDraw object for measuring text
        
    Returns:
        List of text lines
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def cv2_to_pil(img):
    """Convert OpenCV BGR image to PIL RGBA image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb).convert('RGBA')


def pil_to_cv2(pil_img):
    """Convert PIL RGBA image to OpenCV BGR image."""
    img_rgb = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def calculate_fade_alpha(frame_idx, caption_start_frame, caption_end_frame, fade_frames=10):
    """
    Calculate fade-in/fade-out alpha value for smooth caption transitions.
    
    Args:
        frame_idx: Current frame index
        caption_start_frame: First frame where this caption appears
        caption_end_frame: Last frame where this caption appears
        fade_frames: Number of frames for fade transition (default 10)
        
    Returns:
        Alpha value between 0.0 and 1.0
    """
    # Fade in at the start
    if frame_idx < caption_start_frame + fade_frames:
        return (frame_idx - caption_start_frame) / fade_frames
    
    # Fade out at the end
    elif frame_idx > caption_end_frame - fade_frames:
        return (caption_end_frame - frame_idx) / fade_frames
    
    # Full opacity in the middle
    return 1.0


def draw_caption(img, caption_data, alpha=1.0, font_size_title=70, font_size_desc=55, 
                font_size_frame=45, padding=20, corner_radius=15, position='top-left'):
    """
    Draw caption with rounded corners and multi-line layout using PIL.
    
    Args:
        img: OpenCV BGR image to draw on (modified in-place)
        caption_data: Dict with 'desc', 'start', 'end' keys
        alpha: Opacity for fade effects (0.0 to 1.0)
        font_size_title: Font size for skill name
        font_size_desc: Font size for description
        font_size_frame: Font size for frame duration
        padding: Padding inside the caption box
        corner_radius: Radius for rounded corners
        position: Position of caption box ('top-left', 'bottom-left', etc.)
    """
    if alpha <= 0.0:
        return
    
    h, w = img.shape[:2]
    
    # Load fonts (use default if system fonts not available)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_title)
        font_desc = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_desc)
        font_frame = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_frame)
    except:
        # Fallback to default font
        font_title = ImageFont.load_default()
        font_desc = ImageFont.load_default()
        font_frame = ImageFont.load_default()
    
    # Create PIL image for drawing
    pil_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    
    # Prepare text content
    skill_text = f"Skill: {caption_data.get('caption', 'Unknown')}"
    desc_text = caption_data.get('desc', '')
    
    # Two lines for frame duration - check if extracted frame info is available
    # if 'extracted_start' in caption_data and 'extracted_end' in caption_data:
    #     frame_text_1 = f"Extracted Frames (30fps): {caption_data['extracted_start']}-{caption_data['extracted_end']}"
    #     frame_text_2 = f"Original Video Frames (50fps): {int(caption_data['start'])}-{int(caption_data['end'])}"
    # else:
    #     # Fallback to single line if extracted frame info not available
    #     frame_text_1 = f"Frame Duration: {int(caption_data['start'])}-{int(caption_data['end'])}"
    #     frame_text_2 = None
    frame_text_1 = None
    frame_text_2 = None
    
    # Calculate box dimensions - occupy top 1/4 of the frame
    box_height = h // 4  # 1/4 of frame height
    max_text_width = w - 4 * padding - 40  # Leave margin from edges
    
    # Wrap description text
    desc_lines = wrap_text_pil(desc_text, font_desc, max_text_width, draw)
    
    # Calculate box dimensions
    title_bbox = draw.textbbox((0, 0), skill_text, font=font_title)
    title_height = title_bbox[3] - title_bbox[1]
    
    desc_height = 0
    for line in desc_lines:
        bbox = draw.textbbox((0, 0), line, font=font_desc)
        desc_height += (bbox[3] - bbox[1]) + 5  # 5px line spacing
    
    # frame_bbox_1 = draw.textbbox((0, 0), frame_text_1, font=font_frame)
    # frame_height = frame_bbox_1[3] - frame_bbox_1[1]
    # if frame_text_2:
    #     frame_bbox_2 = draw.textbbox((0, 0), frame_text_2, font=font_frame)
    #     frame_height += (frame_bbox_2[3] - frame_bbox_2[1]) + 5  # Add second line height with spacing
    frame_height = 0
    
    box_width = w - 40  # Full width minus margins (20px on each side)
    
    # Position the box - with margin from edges
    if position == 'top-left':
        box_x = 20
        box_y = 20
    elif position == 'bottom-left':
        box_x = 20
        box_y = h - box_height - 20
    else:  # default to top-left
        box_x = 20
        box_y = 20
    
    # Ensure box stays within frame bounds
    if box_x + box_width > w:
        box_width = w - box_x
    
    # Draw rounded rectangle background with transparency
    bg_alpha = int(220 * alpha)  # Semi-transparent white background
    draw.rounded_rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        radius=corner_radius,
        fill=(255, 255, 255, bg_alpha)
    )
    
    # Draw text content
    text_x = box_x + padding + 30  # More left padding
    text_y = box_y + padding + 10  # More top padding
    
    # Title (Skill name)
    text_alpha = int(255 * alpha)
    draw.text((text_x, text_y), skill_text, font=font_title, fill=(30, 30, 30, text_alpha))
    text_y += title_height + padding
    
    # Description lines
    for line in desc_lines:
        draw.text((text_x, text_y), line, font=font_desc, fill=(50, 50, 50, text_alpha))
        bbox = draw.textbbox((0, 0), line, font=font_desc)
        text_y += (bbox[3] - bbox[1]) + 5
    
    text_y += 5  # Extra spacing
    
    # Frame duration (two lines)
    # draw.text((text_x, text_y), frame_text_1, font=font_frame, fill=(80, 80, 80, text_alpha))
    # if frame_text_2:
    #     text_y += frame_bbox_1[3] - frame_bbox_1[1] + 5  # Move to next line
    #     draw.text((text_x, text_y), frame_text_2, font=font_frame, fill=(80, 80, 80, text_alpha))
    
    # Composite PIL image onto OpenCV image
    img_pil_base = cv2_to_pil(img)
    img_composited = Image.alpha_composite(img_pil_base, pil_img)
    img_result = pil_to_cv2(img_composited)
    
    # Copy result back to original image
    img[:] = img_result


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
                       captions_fps=50, start_frame=None, end_frame=None):
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
        start_frame: Start frame index for rendering (optional, 0-based)
        end_frame: End frame index for rendering (optional, exclusive)
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
    # if captions_json and segment_def_json:
    if captions_json:
        task_name = task_path.name  # e.g., '00_task'
        clips = load_captions(captions_json, segment_def_json, task_name)
        if clips:
            print(f"Loaded {len(clips)} caption clips for {task_name}")

    # Load images
    img_dir = task_path / 'extracted_images'
    all_img_files = natsorted(glob(str(img_dir / '*.jpg'))) or natsorted(glob(str(img_dir / '*.png')))
    
    if not all_img_files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    
    # Filter by frame range if specified
    if start_frame is not None or end_frame is not None:
        start_idx = start_frame if start_frame is not None else 0
        end_idx = end_frame if end_frame is not None else len(all_img_files)
        img_files = all_img_files[start_idx:end_idx]
        frame_offset = start_idx
    else:
        img_files = all_img_files
        frame_offset = 0
    
    if not img_files:
        raise ValueError(f"No images in specified range [{start_frame}, {end_frame})")

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
        '-loglevel', 'error',  # Minimize stderr output to prevent pipe deadlock
        '-stats',  # Show encoding progress
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
    
    # Start background thread to drain stderr and prevent pipe deadlock
    stderr_output = []
    stderr_thread = threading.Thread(target=drain_stderr_pipe, 
                                    args=(proc.stderr, stderr_output))
    stderr_thread.daemon = True
    stderr_thread.start()

    try:
        # Track caption changes for fade effects
        prev_caption_id = None
        caption_start_frame = 0
        caption_end_frame = 0
        fade_frames = 10  # Number of frames for fade transition
        
        # Process frames with progress bar
        for local_idx in tqdm(range(len(img_files)), desc="Encoding video"):
            # Check if FFmpeg process is still alive every 100 frames
            if local_idx % 100 == 0 and proc.poll() is not None:
                raise RuntimeError(f"FFmpeg process died unexpectedly with return code {proc.returncode}")
            
            img = cv2.imread(img_files[local_idx])
            if img is None:
                continue
            
            # Calculate actual frame index in original sequence
            frame_idx = local_idx + frame_offset

            # Draw left hand (darker red)
            if frame_idx in left_data:
                draw_hand_skeleton(img, left_data[frame_idx], color_left, thickness=4)

            # Draw right hand (darker green)
            if frame_idx in right_data:
                draw_hand_skeleton(img, right_data[frame_idx], color_right, thickness=4)

            # Draw caption with fade effects if available
            if clips:
                caption_data = get_caption_for_frame(frame_idx, clips, 
                                                    extraction_fps=fps, 
                                                    captions_fps=captions_fps)
                if caption_data:
                    # Create unique ID for this caption based on start/end frames
                    caption_id = (caption_data['start'], caption_data['end'])
                    
                    # Detect caption change
                    if caption_id != prev_caption_id:
                        prev_caption_id = caption_id
                        # Convert caption frame range to extraction fps
                        caption_start_frame = int(caption_data['start'] * fps / captions_fps)
                        caption_end_frame = int(caption_data['end'] * fps / captions_fps)
                    
                    # Calculate fade alpha
                    alpha = calculate_fade_alpha(frame_idx, caption_start_frame, 
                                                caption_end_frame, fade_frames)
                    
                    # Draw caption with calculated alpha
                    draw_caption(img, caption_data, alpha=alpha)

            # Write frame to ffmpeg stdin
            proc.stdin.write(img.tobytes())

        # Close stdin to signal end of input
        proc.stdin.close()

        # Wait for ffmpeg to finish
        proc.wait()
        
        # Wait for stderr thread to finish draining (with timeout)
        stderr_thread.join(timeout=10)

        if proc.returncode == 0:
            print(f"✓ Video saved: {output_path}")
        else:
            stderr = ''.join(stderr_output)
            print(f"✗ FFmpeg error (return code {proc.returncode}):\n{stderr}")

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
    parser.add_argument('--output-dir-per-clip', type=str, default=None,
                       help='Output directory for per-clip videos. When set, renders each caption clip as a separate video.')
    
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

    # Always render the main full video first
    print("Rendering main full video...")
    visualize_to_video(args.dir, args.output, fps=args.fps, device=args.device,
                      captions_json=args.captions_json,
                      segment_def_json=args.segment_def_json,
                      caption_font_scale=args.caption_font_scale,
                      caption_padding=args.caption_padding,
                      captions_fps=args.captions_fps)

    # Optionally render per-clip videos if requested
    if args.output_dir_per_clip:
        if not args.captions_json:
            print("Warning: --output-dir-per-clip requires --captions-json to be specified")
            print("Skipping per-clip rendering.")
        else:
            # Create output directory for per-clip videos
            per_clip_dir = Path(args.output_dir_per_clip)
            per_clip_dir.mkdir(parents=True, exist_ok=True)
            
            # Load captions JSON to get clips
            with open(args.captions_json, 'r') as f:
                captions_data = json.load(f)
            
            # Find matching task in captions
            task_name = task_path.name  # e.g., '00_task'
            task_captions = None
            for video_entry in captions_data:
                video_name = Path(video_entry['video']).stem  # e.g., '00_task' from '00_task.mp4'
                if video_name == task_name:
                    task_captions = video_entry
                    break
            
            if not task_captions or 'clips' not in task_captions:
                print(f"Warning: No clips found for task {task_name} in {args.captions_json}")
                print("Skipping per-clip rendering.")
            else:
                clips = task_captions['clips']
                print(f"\nRendering {len(clips)} clips individually...")
                
                for clip_idx, clip in enumerate(clips):
                    start_frame_caption = clip['start']
                    end_frame_caption = clip['end']
                    
                    # Convert from captions_fps to extraction fps
                    start_frame = int(start_frame_caption * args.fps / args.captions_fps)
                    end_frame = int(end_frame_caption * args.fps / args.captions_fps)
                    
                    # Generate output filename
                    output_filename = f"{task_name}_{start_frame_caption:06d}_{end_frame_caption:06d}.mp4"
                    clip_output_path = per_clip_dir / output_filename
                    
                    print(f"\n[{clip_idx+1}/{len(clips)}] Rendering {output_filename}...")
                    print(f"  Caption frames: [{start_frame_caption}, {end_frame_caption})")
                    print(f"  Extraction frames: [{start_frame}, {end_frame})")
                    
                    # Render this clip
                    try:
                        visualize_to_video(
                            args.dir, clip_output_path, 
                            fps=args.fps, device=args.device,
                            captions_json=args.captions_json,
                            segment_def_json=args.segment_def_json,
                            caption_font_scale=args.caption_font_scale,
                            caption_padding=args.caption_padding,
                            captions_fps=args.captions_fps,
                            start_frame=start_frame,
                            end_frame=end_frame
                        )
                    except Exception as e:
                        print(f"  ✗ Failed to render clip: {e}")
                        continue
                
                print(f"\n✓ All clips rendered to: {per_clip_dir}")


if __name__ == '__main__':
    main()
