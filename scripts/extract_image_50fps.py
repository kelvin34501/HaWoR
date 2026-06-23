# import os

# import numpy as np
# from glob import glob
# from natsort import natsorted
# import subprocess
# import argparse

# def extract_frames(video_path, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     command = [
#         'ffmpeg',               
#         '-i', video_path,       
#         '-vf', 'fps=50',         
#         '-start_number', '0',
#         os.path.join(output_folder, '%04d.jpg')  
#     ]

#     subprocess.run(command, check=True)

# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser(description="Extract")
#     parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
#     parser.add_argument(
#         "--output_folder",
#         type=str,
#         default=None,
#         help="Output folder name",
#     )
#     args = parser.parse_args()
#     extract_frames(args.video_path, args.output_folder)
    
    
import os
from glob import glob
import subprocess
import argparse


def extract_frames(video_path, output_folder, force=False):
    if output_folder is None:
        raise ValueError("output_folder must be provided")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    existing_imgs = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        existing_imgs.extend(glob(os.path.join(output_folder, pattern)))

    if existing_imgs and not force:
        print(f"Skip extracting: output already exists -> {output_folder} ({len(existing_imgs)} files)")
        return

    command = [
        "ffmpeg",
        "-strict", "-2",
        "-i", video_path,
        "-vf", "fps=50",
        "-start_number", "0",
        os.path.join(output_folder, "%04d.jpg")
    ]

    subprocess.run(command, check=True)
    print(f"Saved frames to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extract even if output folder already contains images",
    )
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, force=args.force)