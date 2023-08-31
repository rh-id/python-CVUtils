# This function will be called BEFORE resizing
# image_output_path is output image full path
# frame is the frame image from the video
import cv2


def pre_resize(image_output_path, frame):
    return frame


# This function will be called AFTER resizing
# image_output_path is output image full path
# frame is the frame image from the video
def post_resize(image_output_path, frame):
    return cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
