#!/usr/bin/env python
"""Read the lips in a video file and print the Turkish text.

Usage:
    python scripts/transcribe.py --run exp/my_run video.mp4 [video2.mp4 ...]

The video may be any size and frame rate; it should show one speaker facing
the camera (the tallest face of every frame is taken as the speaker). Long
videos are decoded in pieces of --chunk-seconds, so a word that falls on a
piece boundary can be cut.

Needs MediaPipe and OpenCV besides the training requirements
(pip install mediapipe opencv-python-headless) and assets/face_landmarker.task.

Examples:
    python scripts/transcribe.py --run exp/vsr_finetune --average 10 konusma.mp4
    python scripts/transcribe.py --run exp/vsr_finetune clip.mp4 --save-crop crop.mp4
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from avsr.inference import load_trained, mouth_crops, transcribe  # noqa: E402
from avsr.preprocess import video_io  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoints/")
    parser.add_argument("--average", type=int, default=1, help="Average the weights of the last N checkpoints")
    parser.add_argument("--chunk-seconds", type=float, default=10.0, help="Longest piece decoded at once")
    parser.add_argument("--beam-size", type=int, default=None, help="Default: decode.beam_size of the run")
    parser.add_argument("--save-crop", help="Write the mouth crop of the (first) video here, to check it")
    parser.add_argument("--cpu", action="store_true", help="Decode on the CPU even when a GPU is present")
    parser.add_argument("videos", nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    module, checkpoints, _ = load_trained(args.run, args.average)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    module.to(device).eval()
    print(f"# {len(checkpoints)} checkpoint(s) of {args.run}, decoding on {device}", file=sys.stderr)

    chunk_frames = max(12, int(round(args.chunk_seconds * video_io.VIDEO_FPS)))
    for i, path in enumerate(args.videos):
        crops, detection_rate = mouth_crops(path)
        if detection_rate < 0.8:
            print(f"# {path}: a face was found in only {detection_rate:.0%} of the frames", file=sys.stderr)
        if args.save_crop and i == 0:
            video_io.write_video(args.save_crop, crops)
        pieces = transcribe(module, crops, chunk_frames, args.beam_size)
        if len(pieces) == 1:
            print(f"{path}\t{pieces[0][2]}")
        else:
            for start, end, text in pieces:
                print(f"{path}\t{start:.1f}-{end:.1f}\t{text}")


if __name__ == "__main__":
    main()
