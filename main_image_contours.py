import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("contour_type", choices=['approximation', 'hu_moments'],
                        help="Type of contour analysis that you want to perform")
    parser.add_argument("--export-dir", help="Path to export the images")
    args = parser.parse_args()
    if args.contour_type == 'approximation':
        image_utils.contours_approximation(args.image_path, export_dir=args.export_dir)
    elif args.contour_type == 'hu_moments':
        image_utils.contours_hu_moments(args.image_path, export_dir=args.export_dir)
