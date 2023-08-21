import argparse

from scripts import video_utils
from scripts.model.video_utils import Video2ImageAttr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_path", help="Output image path")
    parser.add_argument("--resize-width", type=int, help="Resized width")
    parser.add_argument("--resize-height", type=int, help="Resized height")
    args = parser.parse_args()
    video_utils.video2image(args.video_path, args.output_path, Video2ImageAttr(args))
