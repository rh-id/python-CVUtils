# CVUtils (Computer Vision Utils)
Python video and image manipulation utilities mainly using opencv

## Main Scripts
File with pattern `main_xxxx` were executable scripts for scripting usage

### main_image_analysis.py
Used to perform image analysis on the input image

### main_image_contours.py
Used to find image contours on the input image

### main_image_feature_matching.py
Used to match image feature

### main_image_histogram_color.py
Used to show input image histogram on color

### main_image_histogram_gray.py
Used to show input image histogram on gray color

### main_image_threshold.py
Used to show input image threshold

### main_pascalvoc_mediapipe_rename.py
Used to rename image in Pascal VOC dataset.
PASCAL VOC based on mediapipe dataset structure where dataset directory consist of `images` and `Annotations` see:
https://developers.google.com/mediapipe/api/solutions/python/mediapipe_model_maker/object_detector/Dataset
 

### main_video2image.py
Used to split video to images.

Use `--filter-path` to point to your custom filter for each frame.
Example run python with: `main_video2image.py assets/video/test.avi output/video2image --filter-path main_video2image/filter/colormap_autumn.py`

See example on how to write [**filter function**](https://github.com/rh-id/python-CVUtils/blob/master/main_video2image/filter/colormap_autumn.py)

### main_video_info.py
Used to show video attributes/information

### main_video_info_scan.py
Used to scan directory and export video attributes/information

## Attributions
Special thanks to [Mastering OpenCV 4](https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python) ([MIT License](https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/LICENSE))