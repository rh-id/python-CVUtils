import argparse

from scripts import pascalvoc_utils

# PASCAL VOC based on mediapipe dataset structure
# where dataset path consist of "images" and "Annotations"
# https://developers.google.com/mediapipe/api/solutions/python/mediapipe_model_maker/object_detector/Dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to the dataset (directory)")
    parser.add_argument("image_file_name", help="Image file that you want to rename")
    parser.add_argument("new_image_file_name", help="The new image file name")
    args = parser.parse_args()

    pascalvoc_utils.rename_image(args.dataset_path, args.image_file_name, args.new_image_file_name)
