import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("threshold_type", choices=['adaptive', 'adaptive_filter_noise',
                                                   'bgr', 'binary', 'otsu', 'otsu_filter_noise',
                                                   'scikit_image', 'triangle_filter_noise'],
                        help="Threshold type that you want to perform")
    args = parser.parse_args()
    if args.threshold_type == 'adaptive':
        image_utils.threshold_image_adaptive(args.image_path)
    elif args.threshold_type == 'adaptive_filter_noise':
        image_utils.threshold_image_adaptive_filter_noise(args.image_path)
    elif args.threshold_type == 'bgr':
        image_utils.threshold_image_bgr(args.image_path)
    elif args.threshold_type == 'binary':
        image_utils.threshold_image_binary(args.image_path)
    elif args.threshold_type == 'otsu':
        image_utils.threshold_image_otsu(args.image_path)
    elif args.threshold_type == 'otsu_filter_noise':
        image_utils.threshold_image_otsu_filter_noise(args.image_path)
    elif args.threshold_type == 'scikit_image':
        image_utils.threshold_image_scikit_image(args.image_path)
    elif args.threshold_type == 'triangle_filter_noise':
        image_utils.threshold_image_triangle_filter_noise(args.image_path)
