import os.path
import pathlib
from urllib import parse

import cv2
import progressbar
from vidgear.gears import CamGear

from scripts.model.video_utils import Video2ImageAttr, Video2ImageYoutubeAttr


def video2image_youtube(output_path, video2image_attr: Video2ImageYoutubeAttr = Video2ImageYoutubeAttr()):
    video2image_attr.print_self()
    filter_pre_resize = video2image_attr.filter_pre_resize
    filter_post_resize = video2image_attr.filter_post_resize

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    options = {"STREAM_RESOLUTION": video2image_attr.stream_resolution}
    for link in video2image_attr.links:
        link_output = os.path.join(output_path, parse.quote_plus(link))
        pathlib.Path(link_output).mkdir(parents=True, exist_ok=True)
        i = 1
        print("Processing:{} ,Output:{}".format(link, link_output))
        cam_gear = (CamGear(source=link, stream_mode=True,
                            **options)
                    .start())
        while True:
            file_path = pathlib.Path(link_output, str(i) + ".jpg")

            frame = cam_gear.read()
            if frame is None:
                break
            image = frame
            if filter_pre_resize is not None:
                image = filter_pre_resize(file_path, image)
            if video2image_attr.resize_width is not None and video2image_attr.resize_height is not None:
                image = cv2.resize(frame, (video2image_attr.resize_width, video2image_attr.resize_height))
            if filter_post_resize is not None:
                image = filter_post_resize(file_path, image)

            cv2.imwrite(str(file_path), image)
            i += 1

        # safely close video stream
        cam_gear.stop()

    print("Done processing {} links".format(len(video2image_attr.links)))


# Split all video at video_path to images (jpeg) at output_path
def video2image(video_path, output_path, video2image_attr: Video2ImageAttr = Video2ImageAttr()):
    capture = cv2.VideoCapture(video_path)

    if capture.isOpened() is False:
        print("Error opening the camera")
        return

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("CV_CAP_PROP_FRAME_WIDTH : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CV_CAP_PROP_FRAME_COUNT : '{}'".format(frame_count))
    video2image_attr.print_self()
    # -----Validate START
    if video2image_attr.start_frame is not None:
        if video2image_attr.start_frame > frame_count:
            raise Exception("Start frame, must not be more than video total frame " + str(frame_count))
        elif video2image_attr.start_frame < 1:
            raise Exception("Start frame must be at least 1")
    if video2image_attr.end_frame is not None:
        if video2image_attr.end_frame > frame_count:
            raise Exception("End frame, must not be more than video total frame " + str(frame_count))
        elif video2image_attr.end_frame < 1:
            raise Exception("End frame must be at least 1")
        elif video2image_attr.start_frame is not None:
            if video2image_attr.start_frame > video2image_attr.end_frame:
                raise Exception("Start frame must not be more than End frame")
    # -----Validate END
    filter_pre_resize = video2image_attr.filter_pre_resize
    filter_post_resize = video2image_attr.filter_post_resize

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    progress_max_val = frame_count
    start_frame = 0
    if video2image_attr.start_frame is not None and video2image_attr.end_frame is not None:
        start_frame = video2image_attr.start_frame - 1
        progress_max_val = video2image_attr.end_frame - start_frame
    elif video2image_attr.start_frame is not None and video2image_attr.end_frame is None:
        start_frame = video2image_attr.start_frame - 1
        progress_max_val = frame_count - start_frame
    elif video2image_attr.start_frame is None and video2image_attr.end_frame is not None:
        progress_max_val = video2image_attr.end_frame

    i = 1
    progress = progressbar.ProgressBar(maxval=progress_max_val).start()
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while capture.isOpened():
        file_path = pathlib.Path(output_path, str(start_frame + 1) + ".jpg")
        ret, frame = capture.read()

        if ret is True:
            image = frame
            if filter_pre_resize is not None:
                image = filter_pre_resize(file_path, image)
            if video2image_attr.resize_width is not None and video2image_attr.resize_height is not None:
                image = cv2.resize(frame, (video2image_attr.resize_width, video2image_attr.resize_height))
            if filter_post_resize is not None:
                image = filter_post_resize(file_path, image)

            cv2.imwrite(str(file_path), image)
            progress.update(i)
            i += 1
            start_frame += 1
            if i > progress_max_val:
                break
        else:
            break

    capture.release()


def get_video_info(video_path):
    capture = cv2.VideoCapture(video_path)
    info = [
        "CV_CAP_PROP_FRAME_WIDTH : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)),
        "CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)),
        "CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)),
        "CAP_PROP_FOURCC  : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))),
        "CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)),
        "CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)),
        "CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)),
        "CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)),
        "CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)),
        "CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)),
        "CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)),
        "CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)),
        "CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)),
        "CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)),
        "CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)),
    ]
    capture.release()
    return info


def print_video_info(video_path):
    info = get_video_info(video_path)
    for i in info:
        print(i)


def decode_fourcc(fourcc):
    fourcc_int = int(fourcc)
    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        fourcc_decode += chr(int_value)
    return fourcc_decode
