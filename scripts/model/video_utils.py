class Video2ImageAttr:
    resize_width: int
    resize_height: int

    def __init__(self, args=None):
        if args is None:
            return
        self.resize_width = args.resize_width
        self.resize_height = args.resize_height
        self.start_frame = args.start_frame
        self.end_frame = args.end_frame

    def print_self(self):
        print("resize_width=" + str(self.resize_width))
        print("resize_height=" + str(self.resize_height))
        print("start_frame=" + str(self.start_frame))
        print("end_frame=" + str(self.end_frame))
