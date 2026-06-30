import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import numpy as np
from lib.pipeline.tools import detect_track
from lib.pipeline.frame_source import frame_source_from_args


def detect_track_video(args):
    file = args.video_path
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = args.seq_dir if getattr(args, 'seq_dir', None) else f'{root}/{seq}'
    os.makedirs(seq_folder, exist_ok=True)
    print(f'Running detect_track on {file} ...')

    ##### Decode frames on demand (no JPEG dump to disk) #####
    frame_source = frame_source_from_args(args, color='bgr')

    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(frame_source)

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, frame_source
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    # boxes_, tracks_ = detect_track(frame_source, thresh=0.2)
    boxes_, tracks_ = detect_track(frame_source, thresh=0.3)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, frame_source

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    args = parser.parse_args()

    detect_track_video(args)