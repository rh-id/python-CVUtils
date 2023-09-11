import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--option", choices=['equalize', 'mask'],
                        help="Extra option to show other type of histogram")
    parser.add_argument("--export-dir", help="Path to export the images")
    args = parser.parse_args()
    if args.option == 'equalize':
        image_utils.histogram_image_gray_equalize(args.image_path, export_dir=args.export_dir)
    elif args.option == 'mask':
        image_utils.histogram_image_gray_mask(args.image_path, export_dir=args.export_dir)
    else:
        image_utils.histogram_image_gray(args.image_path, export_dir=args.export_dir)
