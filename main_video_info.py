import argparse
import video_utils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()
    video_utils.print_video_info(args.video_path)
