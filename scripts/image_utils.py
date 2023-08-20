import cv2
import numpy as np
from matplotlib import pyplot as plt


def analyze_image_color_map(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(4, 7, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    colormaps = ['AUTUMN', 'BONE', 'JET', 'WINTER', 'RAINBOW', 'OCEAN',
                 'SUMMER', 'SPRING', 'COOL', 'HSV', 'PINK', 'HOT',
                 'PARULA', 'MAGMA', 'INFERNO', 'PLASMA', 'VIRIDIS', 'CIVIDIS',
                 'TWILIGHT', 'TWILIGHT_SHIFTED', 'TURBO', 'DEEPGREEN']

    plt.figure(figsize=(12, 5))
    plt.suptitle("Colormaps", fontsize=14, fontweight='bold')

    show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "GRAY", 1)

    for idx, val in enumerate(colormaps):
        show_with_matplotlib(cv2.applyColorMap(gray_img, idx), val, idx + 2)

    plt.show()


# Analyze image with color map
def analyze_image_color_space(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 6, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    image = cv2.imread(image_path)

    plt.figure(figsize=(12, 5))
    plt.suptitle("Color spaces in OpenCV", fontsize=14, fontweight='bold')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (bgr_b, bgr_g, bgr_r) = cv2.split(image)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (hsv_h, hsv_s, hsv_v) = cv2.split(hsv_image)

    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    (hls_h, hls_l, hls_s) = cv2.split(hls_image)

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    (ycrcb_y, ycrcb_cr, ycrcb_cb) = cv2.split(ycrcb_image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    (lab_l, lab_a, lab_b) = cv2.split(lab_image)

    show_with_matplotlib(image, "BGR - image", 1)

    show_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray image", 1 + 6)

    show_with_matplotlib(cv2.cvtColor(bgr_b, cv2.COLOR_GRAY2BGR), "BGR - B comp", 2)
    show_with_matplotlib(cv2.cvtColor(bgr_g, cv2.COLOR_GRAY2BGR), "BGR - G comp", 2 + 6)
    show_with_matplotlib(cv2.cvtColor(bgr_r, cv2.COLOR_GRAY2BGR), "BGR - R comp", 2 + 6 * 2)

    show_with_matplotlib(cv2.cvtColor(hsv_h, cv2.COLOR_GRAY2BGR), "HSV - H comp", 3)
    show_with_matplotlib(cv2.cvtColor(hsv_s, cv2.COLOR_GRAY2BGR), "HSV - S comp", 3 + 6)
    show_with_matplotlib(cv2.cvtColor(hsv_v, cv2.COLOR_GRAY2BGR), "HSV - V comp", 3 + 6 * 2)

    show_with_matplotlib(cv2.cvtColor(hls_h, cv2.COLOR_GRAY2BGR), "HLS - H comp", 4)
    show_with_matplotlib(cv2.cvtColor(hls_l, cv2.COLOR_GRAY2BGR), "HLS - L comp", 4 + 6)
    show_with_matplotlib(cv2.cvtColor(hls_s, cv2.COLOR_GRAY2BGR), "HLS - S comp", 4 + 6 * 2)

    show_with_matplotlib(cv2.cvtColor(ycrcb_y, cv2.COLOR_GRAY2BGR), "YCrCb - Y comp", 5)
    show_with_matplotlib(cv2.cvtColor(ycrcb_cr, cv2.COLOR_GRAY2BGR), "YCrCb - Cr comp", 5 + 6)
    show_with_matplotlib(cv2.cvtColor(ycrcb_cb, cv2.COLOR_GRAY2BGR), "YCrCb - Cb comp", 5 + 6 * 2)

    show_with_matplotlib(cv2.cvtColor(lab_l, cv2.COLOR_GRAY2BGR), "L*a*b - L comp", 6)
    show_with_matplotlib(cv2.cvtColor(lab_a, cv2.COLOR_GRAY2BGR), "L*a*b - a comp", 6 + 6)
    show_with_matplotlib(cv2.cvtColor(lab_b, cv2.COLOR_GRAY2BGR), "L*a*b - b comp", 6 + 6 * 2)

    plt.show()


# Analyze image with color split
def analyze_image_color_split(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 6, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    image = cv2.imread(image_path)

    plt.figure(figsize=(14, 6))
    plt.suptitle("Splitting and merging channels in OpenCV", fontsize=14, fontweight='bold')

    show_with_matplotlib(image, "BGR - image", 1)

    (b, g, r) = cv2.split(image)

    show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR - (B)", 2)
    show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR - (G)", 2 + 6)
    show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR - (R)", 2 + 6 * 2)

    image_copy = cv2.merge((b, g, r))

    show_with_matplotlib(image_copy, "BGR - image (copy)", 1 + 6)

    image_without_blue = image.copy()
    image_without_blue[:, :, 0] = 0
    image_without_green = image.copy()
    image_without_green[:, :, 1] = 0
    image_without_red = image.copy()
    image_without_red[:, :, 2] = 0

    show_with_matplotlib(image_without_blue, "BGR without B", 3)
    show_with_matplotlib(image_without_green, "BGR without G", 3 + 6)
    show_with_matplotlib(image_without_red, "BGR without R", 3 + 6 * 2)

    (b, g, r) = cv2.split(image_without_blue)

    show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without B (B)", 4)
    show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without B (G)", 4 + 6)
    show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without B (R)", 4 + 6 * 2)

    (b, g, r) = cv2.split(image_without_green)

    show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without G (B)", 5)
    show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without G (G)", 5 + 6)
    show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without G (R)", 5 + 6 * 2)

    (b, g, r) = cv2.split(image_without_red)

    show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 6)
    show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 6 + 6)
    show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 6 + 6 * 2)

    plt.show()


# Analyze image with multiple sharpen tech
def analyze_image_sharpen(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def unsharped_filter(img):
        """The unsharp filter enhances edges subtracting the smoothed image from the original image"""

        smoothed = cv2.GaussianBlur(img, (9, 9), 10)
        return cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Sharpening images", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)

    kernel_sharpen_1 = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

    kernel_sharpen_2 = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])

    kernel_sharpen_3 = np.array([[1, 1, 1],
                                 [1, -7, 1],
                                 [1, 1, 1]])

    kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0

    sharp_image_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
    sharp_image_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
    sharp_image_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
    sharp_image_4 = cv2.filter2D(image, -1, kernel_sharpen_4)

    sharp_image_5 = unsharped_filter(image)

    print("kernel_sharpen_1: {}".format(kernel_sharpen_1))
    print("kernel_sharpen_2: {}".format(kernel_sharpen_2))
    print("kernel_sharpen_3: {}".format(kernel_sharpen_3))
    print("kernel_sharpen_4: {}".format(kernel_sharpen_4))

    show_with_matplotlib(image, "original", 1)
    show_with_matplotlib(sharp_image_1, "sharp 1", 2)
    show_with_matplotlib(sharp_image_2, "sharp 2", 3)
    show_with_matplotlib(sharp_image_3, "sharp 3", 4)
    show_with_matplotlib(sharp_image_4, "sharp 4", 5)
    show_with_matplotlib(sharp_image_5, "sharp 5", 6)

    plt.show()


# Analyze image with multiple smooth tech
def analyze_image_smooth(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    plt.figure(figsize=(12, 6))
    plt.suptitle("Smoothing techniques", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)

    kernel_averaging_10_10 = np.ones((10, 10), np.float32) / 100

    kernel_averaging_5_5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                                     [0.04, 0.04, 0.04, 0.04, 0.04],
                                     [0.04, 0.04, 0.04, 0.04, 0.04],
                                     [0.04, 0.04, 0.04, 0.04, 0.04],
                                     [0.04, 0.04, 0.04, 0.04, 0.04]])

    print("kernel_5_5: {}".format(kernel_averaging_5_5))
    print("kernel_10_10: {}".format(kernel_averaging_10_10))

    smooth_image_f2D_5_5 = cv2.filter2D(image, -1, kernel_averaging_5_5)
    smooth_image_f2D_10_10 = cv2.filter2D(image, -1, kernel_averaging_10_10)

    smooth_image_b = cv2.blur(image, (10, 10))

    smooth_image_bfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)

    smooth_image_gb = cv2.GaussianBlur(image, (9, 9), 0)

    smooth_image_mb = cv2.medianBlur(image, 9)

    smooth_image_bf = cv2.bilateralFilter(image, 5, 10, 10)
    smooth_image_bf_2 = cv2.bilateralFilter(image, 9, 200, 200)

    show_with_matplotlib(image, "original", 1)
    show_with_matplotlib(smooth_image_f2D_5_5, "cv2.filter2D() (5,5) kernel", 2)
    show_with_matplotlib(smooth_image_f2D_10_10, "cv2.filter2D() (10,10) kernel", 3)
    show_with_matplotlib(smooth_image_b, "cv2.blur()", 4)
    show_with_matplotlib(smooth_image_bfi, "cv2.boxFilter()", 5)
    show_with_matplotlib(smooth_image_gb, "cv2.GaussianBlur()", 6)
    show_with_matplotlib(smooth_image_mb, "cv2.medianBlur()", 7)
    show_with_matplotlib(smooth_image_bf, "cv2.bilateralFilter() - small values", 8)
    show_with_matplotlib(smooth_image_bf_2, "cv2.bilateralFilter() - big values", 9)

    plt.show()
