import argparse

from scripts import video_utils
from scripts.model.video_utils import Video2ImageYoutubeAttr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("links_path", help="Path to text file that contains list of youtube links")
    parser.add_argument("output_path", help="Output image path")
    parser.add_argument("--stream-resolution", type=str,
                        default='best',
                        choices=['144p', '240p', '360p', '480p', '720p', '1080p', '1440p', '2160p', '4320p', 'worst',
                                 'best'],
                        help="Youtube stream resolutions")
    parser.add_argument("--resize-width", type=int, help="Resized width")
    parser.add_argument("--resize-height", type=int, help="Resized height")
    parser.add_argument("--filter-path", type=str, help="Python script that you want to use to filter each frame")
    args = parser.parse_args()
    video_utils.video2image_youtube(args.output_path, Video2ImageYoutubeAttr(args))
