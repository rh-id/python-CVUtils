import argparse
import os
import pathlib

from scripts import video_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video files directory")
    parser.add_argument("info_path", help="Path to the output info directory")
    args = parser.parse_args()
    video_path = args.video_path
    info_path = args.info_path
    if os.path.isfile(video_path):
        raise Exception("video_path must be a directory")
    if os.path.isfile(info_path):
        raise Exception("info_path must be a directory")
    pathlib.Path(info_path).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(video_path):
        f = os.path.join(video_path, filename)
        if os.path.isfile(f):
            info = video_utils.get_video_info(f)
            with open(os.path.join(info_path, filename + '.txt'), 'x') as f:
                for i in info:
                    f.write(i + os.linesep)
