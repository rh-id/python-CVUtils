import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("threshold_type", choices=['binary'],
                        help="Threshold type that you want to perform")
    args = parser.parse_args()
    if args.threshold_type == 'binary':
        image_utils.threshold_image_binary(args.image_path)
