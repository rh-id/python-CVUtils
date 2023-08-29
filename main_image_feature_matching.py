import argparse

from scripts import image_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path_query", help="Image file you want to match")
    parser.add_argument("image_path_scene", help="Image file you want to match to")
    parser.add_argument("matcher", choices=['bf'],
                        help="Image file you want to match to")
    args = parser.parse_args()
    if args.matcher == 'bf':
        image_utils.feature_matching_bf(args.image_path_query, args.image_path_scene)
