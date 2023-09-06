# PASCAL VOC based on mediapipe dataset structure
# where dataset directory consist of "images" and "Annotations"
# https://developers.google.com/mediapipe/api/solutions/python/mediapipe_model_maker/object_detector/Dataset
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def get_annotations_path(dataset_path):
    return os.path.join(dataset_path, 'Annotations')


def get_images_path(dataset_path):
    return os.path.join(dataset_path, 'images')


def get_annotation_xml(dataset_path, image_file_name):
    return os.path.join(get_annotations_path(dataset_path),
                        os.path.splitext(image_file_name)[0] + '.xml')


def get_image(dataset_path, file_name):
    return os.path.join(get_images_path(dataset_path), file_name)


def rename_image(dataset_path, image_file_name, new_image_file_name):
    if not os.path.isdir(dataset_path):
        raise Exception("Dataset path must be a directory")

    # Process Annotation XML file
    annotation_file = get_annotation_xml(dataset_path, image_file_name)
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    x_filename = root.find("filename")
    x_filename.text = new_image_file_name
    x_path = root.find("path")
    old_dir_name = os.path.dirname(x_path.text)
    x_path.text = os.path.join(old_dir_name, new_image_file_name)
    tree.write(annotation_file)

    # Rename Annotation file
    annotation_path = Path(annotation_file)
    annotation_path.rename(annotation_path.with_name(os.path.splitext(new_image_file_name)[0] + '.xml'))

    # Rename image file
    image_file = get_image(dataset_path, image_file_name)
    image_file_path = Path(image_file)
    image_file_path.rename(image_file_path.with_name(new_image_file_name))
