import pathlib

import cv2
import progressbar


# Split all video at video_path to images (jpeg) at output_path
def video2image(video_path, output_path, resize_width=None, resize_height=None):
    capture = cv2.VideoCapture(video_path)

    if capture.isOpened() is False:
        print("Error opening the camera")
        return

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 1
    progress = progressbar.ProgressBar(max_value=frame_count)
    print("CV_CAP_PROP_FRAME_WIDTH : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CV_CAP_PROP_FRAME_COUNT : '{}'".format(frame_count))
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    while capture.isOpened():
        file_path = pathlib.Path(output_path, str(i) + ".jpg")
        ret, frame = capture.read()

        if ret is True:
            image = frame
            if resize_width is not None and resize_height is not None:
                image = cv2.resize(frame, (resize_width, resize_height))

            cv2.imwrite(str(file_path), image)
            progress.update(i)
            i += 1
        else:
            break

    capture.release()


def print_video_info(video_path):
    capture = cv2.VideoCapture(video_path)
    print("CV_CAP_PROP_FRAME_WIDTH : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
    print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
    print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    print("CAP_PROP_FOURCC  : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
    print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
    print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
    print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
    print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
    print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
    print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
    print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
    print("CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
    print("CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
    print("CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))
    capture.release()


def decode_fourcc(fourcc):
    fourcc_int = int(fourcc)
    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        fourcc_decode += chr(int_value)
    return fourcc_decode
