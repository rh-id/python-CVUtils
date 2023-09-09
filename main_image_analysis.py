import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("analyze_type", choices=['kernel', 'morph', 'sharpen', 'skin_segment',
                                                 'smooth', 'color_map', 'color_space',
                                                 'color_split'],
                        help="Type of image analysis that you want to perform")
    parser.add_argument("--export-dir", help="Path to export the images")
    args = parser.parse_args()
    if args.analyze_type == 'kernel':
        image_utils.analyze_image_kernel(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'morph':
        image_utils.analyze_image_morph(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'sharpen':
        image_utils.analyze_image_sharpen(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'skin_segment':
        image_utils.analyze_image_skin_segment(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'smooth':
        image_utils.analyze_image_smooth(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'color_map':
        image_utils.analyze_image_color_map(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'color_split':
        image_utils.analyze_image_color_split(args.image_path, export_dir=args.export_dir)
    elif args.analyze_type == 'color_space':
        image_utils.analyze_image_color_space(args.image_path, export_dir=args.export_dir)
