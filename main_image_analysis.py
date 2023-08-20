import argparse

from scripts import image_utils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("analyze_type", choices=['smooth', 'sharpen', 'color_split'],
                        help="Type of image analysis that you want to perform")
    args = parser.parse_args()
    # TODO color_space,filter2d_kernels,morph,skin_segment
    if args.analyze_type == 'smooth':
        image_utils.analyze_image_smooth(args.image_path)
    elif args.analyze_type == 'sharpen':
        image_utils.analyze_image_sharpen(args.image_path)
    elif args.analyze_type == 'color_split':
        image_utils.analyze_image_color_split(args.image_path)
