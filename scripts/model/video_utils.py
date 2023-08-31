import os
import sys
from importlib import util


class Video2ImageAttr:
    resize_width: int
    resize_height: int
    start_frame: int
    end_frame: int

    filter_path: str
    filter_module_name: str
    filter_pre_resize: any
    filter_post_resize: any

    def __init__(self, args=None):
        if args is None:
            return
        self.resize_width = args.resize_width
        self.resize_height = args.resize_height
        self.start_frame = args.start_frame
        self.end_frame = args.end_frame
        self.filter_path = args.filter_path
        if self.filter_path is not None:
            self.filter_module_name = os.path.basename(self.filter_path).rsplit('.')[0]
            spec = util.spec_from_file_location(self.filter_module_name,
                                                self.filter_path)
            mod = util.module_from_spec(spec)
            sys.modules[self.filter_module_name] = mod
            spec.loader.exec_module(mod)
            self.filter_pre_resize = getattr(mod, 'pre_resize')
            self.filter_post_resize = getattr(mod, 'post_resize')
        else:
            self.filter_pre_resize = None
            self.filter_post_resize = None

    def print_self(self):
        print("resize_width=" + str(self.resize_width))
        print("resize_height=" + str(self.resize_height))
        print("start_frame=" + str(self.start_frame))
        print("end_frame=" + str(self.end_frame))
        print("filter_path=" + str(self.filter_path))
        print("filter_module_name=" + str(self.filter_path))
        if self.filter_pre_resize is None:
            print("no filter_pre_resize function")
        else:
            print("filter_pre_resize=" + str(self.filter_pre_resize))
        if self.filter_post_resize is None:
            print("no filter_post_resize function")
        else:
            print("filter_post_resize=" + str(self.filter_post_resize))
