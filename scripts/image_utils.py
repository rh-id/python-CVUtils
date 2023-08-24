import cv2
import numpy as np
from matplotlib import pyplot as plt


def threshold_image_adaptive_filter_noise(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(15, 7))
    plt.suptitle("Thresholding Adaptive + Bilateral Filter (noise removal while edges sharp)", fontsize=14,
                 fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)

    thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "GAUSSIAN_C, blockSize=31, C=3", 6)

    plt.show()


def threshold_image_adaptive(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(15, 7))
    plt.suptitle("Thresholding Adaptive", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "GAUSSIAN_C, blockSize=31, C=3", 6)

    plt.show()


def threshold_image_binary(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("Thresholding Binary", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

    ret1, thresh1 = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
    ret4, thresh4 = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
    ret5, thresh5 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    ret6, thresh6 = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
    ret7, thresh7 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
    ret8, thresh8 = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)

    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 60", 2)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 70", 3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 80", 4)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 90", 5)
    show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 100", 6)
    show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 110", 7)
    show_img_with_matplotlib(cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "threshold = 120", 8)
    show_img_with_matplotlib(cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR), "threshold = 130", 9)

    plt.show()


def histogram_image_color_equalize_hsv(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_rgb(hist, pos, color):
        plt.subplot(3, 4, pos)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])

        for (h, c) in zip(hist, color):
            plt.plot(h, color=c)

    def hist_color_img(img):
        histr = []
        histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
        return histr

    def equalize_hist_color_hsv(img):
        H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
        return eq_image

    plt.figure(figsize=(18, 14))
    plt.suptitle("Color histogram equalization with cv2.equalizeHist() in the V channel", fontsize=14,
                 fontweight='bold')

    image = cv2.imread(image_path)

    hist_color = hist_color_img(image)

    image_eq = equalize_hist_color_hsv(image)
    hist_image_eq = hist_color_img(image_eq)

    # Add/Subtract 15 to every pixel on the image and calculate histogram
    M = np.ones(image.shape, dtype="uint8") * 15
    added_image = cv2.add(image, M)
    hist_color_added_image = hist_color_img(added_image)

    added_image_eq = equalize_hist_color_hsv(added_image)
    hist_added_image_eq = hist_color_img(added_image_eq)

    subtracted_image = cv2.subtract(image, M)
    hist_color_subtracted_image = hist_color_img(subtracted_image)

    subtracted_image_eq = equalize_hist_color_hsv(subtracted_image)
    hist_subtracted_image_eq = hist_color_img(subtracted_image_eq)

    show_img_with_matplotlib(image, "image", 1)
    show_hist_with_matplotlib_rgb(hist_color, 2, ['b', 'g', 'r'])
    show_img_with_matplotlib(added_image, "image lighter", 5)
    show_hist_with_matplotlib_rgb(hist_color_added_image, 6, ['b', 'g', 'r'])
    show_img_with_matplotlib(subtracted_image, "image darker", 9)
    show_hist_with_matplotlib_rgb(hist_color_subtracted_image, 10, ['b', 'g', 'r'])

    show_img_with_matplotlib(image_eq, "image equalized", 3)
    show_hist_with_matplotlib_rgb(hist_image_eq, 4, ['b', 'g', 'r'])
    show_img_with_matplotlib(added_image_eq, "image lighter equalized", 7)
    show_hist_with_matplotlib_rgb(hist_added_image_eq, 8, ['b', 'g', 'r'])
    show_img_with_matplotlib(subtracted_image_eq, "image darker equalized", 11)
    show_hist_with_matplotlib_rgb(hist_subtracted_image_eq, 12, ['b', 'g', 'r'])

    plt.show()


def histogram_image_color_equalize(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_rgb(hist, pos, color):
        plt.subplot(3, 4, pos)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])

        for (h, c) in zip(hist, color):
            plt.plot(h, color=c)

    def hist_color_img(img):
        histr = []
        histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
        return histr

    def equalize_hist_color(img):
        channels = cv2.split(img)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cv2.equalizeHist(ch))

        eq_image = cv2.merge(eq_channels)
        return eq_image

    plt.figure(figsize=(18, 14))
    plt.suptitle("Color histogram equalization with cv2.equalizeHist() - not a good approach", fontsize=14,
                 fontweight='bold')

    image = cv2.imread(image_path)

    hist_color = hist_color_img(image)

    image_eq = equalize_hist_color(image)
    hist_image_eq = hist_color_img(image_eq)

    # Add/Subtract 15 to every pixel on the image and calculate histogram
    M = np.ones(image.shape, dtype="uint8") * 15
    added_image = cv2.add(image, M)
    hist_color_added_image = hist_color_img(added_image)

    added_image_eq = equalize_hist_color(added_image)
    hist_added_image_eq = hist_color_img(added_image_eq)

    subtracted_image = cv2.subtract(image, M)
    hist_color_subtracted_image = hist_color_img(subtracted_image)

    subtracted_image_eq = equalize_hist_color(subtracted_image)
    hist_subtracted_image_eq = hist_color_img(subtracted_image_eq)

    show_img_with_matplotlib(image, "image", 1)
    show_hist_with_matplotlib_rgb(hist_color, 2, ['b', 'g', 'r'])
    show_img_with_matplotlib(added_image, "image lighter", 5)
    show_hist_with_matplotlib_rgb(hist_color_added_image, 6, ['b', 'g', 'r'])
    show_img_with_matplotlib(subtracted_image, "image darker", 9)
    show_hist_with_matplotlib_rgb(hist_color_subtracted_image, 10, ['b', 'g', 'r'])

    show_img_with_matplotlib(image_eq, "image equalized", 3)
    show_hist_with_matplotlib_rgb(hist_image_eq, 4, ['b', 'g', 'r'])
    show_img_with_matplotlib(added_image_eq, "image lighter equalized", 7)
    show_hist_with_matplotlib_rgb(hist_added_image_eq, 8, ['b', 'g', 'r'])
    show_img_with_matplotlib(subtracted_image_eq, "image darker equalized", 11)
    show_hist_with_matplotlib_rgb(hist_subtracted_image_eq, 12, ['b', 'g', 'r'])

    plt.show()


def histogram_image_color(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_rgb(hist, pos, color):
        plt.subplot(2, 3, pos)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])

        for (h, c) in zip(hist, color):
            plt.plot(h, color=c)

    def hist_color_img(img):
        histr = []
        histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
        histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
        return histr

    plt.figure(figsize=(15, 6))
    plt.suptitle("Color histograms", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)

    hist_color = hist_color_img(image)

    show_img_with_matplotlib(image, "image", 1)

    show_hist_with_matplotlib_rgb(hist_color, 4, ['b', 'g', 'r'])

    # Add/Subtract 15 to every pixel on the image and calculate histogram:
    M = np.ones(image.shape, dtype="uint8") * 15
    added_image = cv2.add(image, M)
    hist_color_added_image = hist_color_img(added_image)

    subtracted_image = cv2.subtract(image, M)
    hist_color_subtracted_image = hist_color_img(subtracted_image)

    show_img_with_matplotlib(added_image, "image lighter", 2)
    show_hist_with_matplotlib_rgb(hist_color_added_image, 5, ['b', 'g', 'r'])
    show_img_with_matplotlib(subtracted_image, "image darker", 3)
    show_hist_with_matplotlib_rgb(hist_color_subtracted_image, 6, ['b', 'g', 'r'])

    plt.show()


def histogram_image_gray_mask(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 2, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, pos, color):
        plt.subplot(2, 2, pos)
        plt.xlabel("bins")
        plt.xlim([0, 256])
        plt.plot(hist, color=color)

    plt.figure(figsize=(10, 6))
    plt.suptitle("Grayscale masked histogram", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram calling cv2.calcHist()
    # The first argument it the list of images to process
    # The second argument is the indexes of the channels to be used to calculate the histogram
    # The third argument is a mask to compute the histogram for the masked pixels
    # The fourth argument is a list containing the number of bins for each channel
    # The fifth argument is the range of possible pixel values
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
    show_hist_with_matplotlib_gray(hist, 2, 'm')

    # Create the mask and calculate the histogram using the mask:
    mask = np.zeros(gray_image.shape[:2], np.uint8)
    mask[30:190, 30:190] = 255
    hist_mask = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])

    masked_img = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    show_img_with_matplotlib(cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR), "masked gray image", 3)
    show_hist_with_matplotlib_gray(hist_mask, 4, 'm')

    plt.show()


def histogram_image_gray_equalize(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, pos, color):
        plt.subplot(3, 4, pos)
        plt.xlabel("bins")
        plt.xlim([0, 256])
        plt.plot(hist, color=color)

    plt.figure(figsize=(20, 16))
    plt.suptitle("Grayscale histogram equalization with cv2.equalizeHist()", fontsize=16, fontweight='bold')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram calling cv2.calcHist()
    # The first argument it the list of images to process
    # The second argument is the indexes of the channels to be used to calculate the histogram
    # The third argument is a mask to compute the histogram for the masked pixels
    # The fourth argument is a list containing the number of bins for each channel
    # The fifth argument is the range of possible pixel values
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    gray_image_eq = cv2.equalizeHist(gray_image)
    hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

    # Add/Substract 35 to every pixel on the grayscale image and calculate histogram:
    M = np.ones(gray_image.shape, dtype="uint8") * 35
    added_image = cv2.add(gray_image, M)
    hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

    added_image_eq = cv2.equalizeHist(added_image)
    hist_eq_added_image = cv2.calcHist([added_image_eq], [0], None, [256], [0, 256])

    subtracted_image = cv2.subtract(gray_image, M)
    hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

    subtracted_image_eq = cv2.equalizeHist(subtracted_image)
    hist_eq_subtracted_image = cv2.calcHist([subtracted_image_eq], [0], None, [256], [0, 256])

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
    show_hist_with_matplotlib_gray(hist, 2, 'm')
    show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
    show_hist_with_matplotlib_gray(hist_added_image, 6, 'm')
    show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 9)
    show_hist_with_matplotlib_gray(hist_subtracted_image, 10, 'm')

    show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3)
    show_hist_with_matplotlib_gray(hist_eq, 4, 'm')
    show_img_with_matplotlib(cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
    show_hist_with_matplotlib_gray(hist_eq_added_image, 8, 'm')
    show_img_with_matplotlib(cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
    show_hist_with_matplotlib_gray(hist_eq_subtracted_image, 12, 'm')

    plt.show()


def histogram_image_gray(image_path):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, pos, color):
        plt.subplot(2, 3, pos)
        plt.xlabel("bins")
        plt.xlim([0, 256])
        plt.plot(hist, color=color)

    plt.figure(figsize=(18, 8))
    plt.suptitle("Grayscale histograms", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
    show_hist_with_matplotlib_gray(hist, 4, 'm')

    M = np.ones(gray_image.shape, dtype="uint8") * 35
    added_image = cv2.add(gray_image, M)
    hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

    subtracted_image = cv2.subtract(gray_image, M)
    hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

    show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 2)
    show_hist_with_matplotlib_gray(hist_added_image, 5, 'm')
    show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 3)
    show_hist_with_matplotlib_gray(hist_subtracted_image, 6, 'm')

    plt.show()


def analyze_image_kernel(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    plt.figure(figsize=(12, 6))
    plt.suptitle("Comparing different kernels using cv2.filter2D()", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)

    kernel_identity = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

    kernel_edge_detection_1 = np.array([[1, 0, -1],
                                        [0, 0, 0],
                                        [-1, 0, 1]])

    kernel_edge_detection_2 = np.array([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]])

    kernel_edge_detection_3 = np.array([[-1, -1, -1],
                                        [-1, 8, -1],
                                        [-1, -1, -1]])

    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    kernel_unsharp_masking = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                                  [4, 16, 24, 16, 4],
                                                  [6, 24, -476, 24, 6],
                                                  [4, 16, 24, 16, 4],
                                                  [1, 4, 6, 4, 1]])

    kernel_blur = 1 / 9 * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

    gaussian_blur = 1 / 16 * np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]])

    kernel_emboss = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])

    sobel_x_kernel = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    outline_kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])

    original_image = cv2.filter2D(image, -1, kernel_identity)
    edge_image_1 = cv2.filter2D(image, -1, kernel_edge_detection_1)
    edge_image_2 = cv2.filter2D(image, -1, kernel_edge_detection_2)
    edge_image_3 = cv2.filter2D(image, -1, kernel_edge_detection_3)
    sharpen_image = cv2.filter2D(image, -1, kernel_sharpen)
    unsharp_masking_image = cv2.filter2D(image, -1, kernel_unsharp_masking)
    blur_image = cv2.filter2D(image, -1, kernel_blur)
    gaussian_blur_image = cv2.filter2D(image, -1, gaussian_blur)
    emboss_image = cv2.filter2D(image, -1, kernel_emboss)
    sobel_x_image = cv2.filter2D(image, -1, sobel_x_kernel)
    sobel_y_image = cv2.filter2D(image, -1, sobel_y_kernel)
    outline_image = cv2.filter2D(image, -1, outline_kernel)

    # Show all the images:
    show_with_matplotlib(original_image, "identity kernel", 1)
    show_with_matplotlib(edge_image_1, "edge detection 1", 2)
    show_with_matplotlib(edge_image_2, "edge detection 2", 3)
    show_with_matplotlib(edge_image_3, "edge detection 3", 4)
    show_with_matplotlib(sharpen_image, "sharpen", 5)
    show_with_matplotlib(unsharp_masking_image, "unsharp masking", 6)
    show_with_matplotlib(blur_image, "blur image", 7)
    show_with_matplotlib(gaussian_blur_image, "gaussian blur image", 8)
    show_with_matplotlib(emboss_image, "emboss image", 9)
    show_with_matplotlib(sobel_x_image, "sobel x image", 10)
    show_with_matplotlib(sobel_y_image, "sobel y image", 11)
    show_with_matplotlib(outline_image, "outline image", 12)

    # Show the Figure:
    plt.show()


def analyze_image_skin_segment(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    lower_hsv = np.array([0, 48, 80], dtype="uint8")
    upper_hsv = np.array([20, 255, 255], dtype="uint8")
    lower_hsv_2 = np.array([0, 50, 0], dtype="uint8")
    upper_hsv_2 = np.array([120, 150, 255], dtype="uint8")
    lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
    upper_ycrcb = np.array([255, 173, 127], dtype="uint8")

    def skin_detector_hsv(bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        skin_region = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        return skin_region

    def skin_detector_hsv_2(bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        skin_region = cv2.inRange(hsv_image, lower_hsv_2, upper_hsv_2)
        return skin_region

    def skin_detector_ycrcb(bgr_image):
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
        skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
        return skin_region

    def bgr_skin(b, g, r):
        """Rule for skin pixel segmentation based on the paper 'RGB-H-CbCr Skin Colour Model for Human Face Detection'"""

        e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
                abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
        e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
        return e1 or e2

    def skin_detector_bgr(bgr_image):
        h = bgr_image.shape[0]
        w = bgr_image.shape[1]
        res = np.zeros((h, w, 1), dtype="uint8")
        # Only 'skin pixels' will be set to white (255) in the res image:
        for y in range(0, h):
            for x in range(0, w):
                (b, g, r) = bgr_image[y, x]
                if bgr_skin(b, g, r):
                    res[y, x] = 255

        return res

    skin_detectors = {
        'ycrcb': skin_detector_ycrcb,
        'hsv': skin_detector_hsv,
        'hsv_2': skin_detector_hsv_2,
        'bgr': skin_detector_bgr
    }

    plt.figure(figsize=(15, 8))
    plt.suptitle("Skin segmentation using different color spaces", fontsize=14, fontweight='bold')

    image = cv2.imread(image_path)
    show_with_matplotlib(image, "Input Image", 1)
    for i, (k, v) in enumerate(skin_detectors.items()):
        detected_skin = v(image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        show_with_matplotlib(bgr, k, i + 2)
    plt.show()


def analyze_image_morph(image_path):
    def show_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def build_kernel(kernel_type, kernel_size):
        if kernel_type == cv2.MORPH_ELLIPSE:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        elif kernel_type == cv2.MORPH_CROSS:
            return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        else:
            return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def erode(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        erosion = cv2.erode(image, kernel, iterations=1)
        return erosion

    def dilate(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        dilation = cv2.dilate(image, kernel, iterations=1)
        return dilation

    # Closing = dilation + erosion
    def closing(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        clos = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return clos

    # Opening = erosion + dilation
    def opening(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        ope = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return ope

    def morphological_gradient(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return morph_gradient

    def top_hat(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        morph = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return morph

    def black_hat(image, kernel_type, kernel_size):
        kernel = build_kernel(kernel_type, kernel_size)
        morph = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return morph

    def closing_and_opening(image, kernel_type, kernel_size):
        closing_img = closing(image, kernel_type, kernel_size)
        opening_img = opening(closing_img, kernel_type, kernel_size)
        return opening_img

    def opening_and_closing(image, kernel_type, kernel_size):

        opening_img = opening(image, kernel_type, kernel_size)
        closing_img = closing(opening_img, kernel_type, kernel_size)
        return closing_img

    morphological_operations = {
        'erode': erode,
        'dilate': dilate,
        'gradient': morphological_gradient,
        'closing': closing,
        'opening': opening,
        'tophat': top_hat,
        'blackhat': black_hat,
        'closing|opening': closing_and_opening,
        'opening|closing': opening_and_closing
    }
    kernel_size_3_3 = (3, 3)
    kernel_size_5_5 = (5, 5)

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(3,3)'", fontsize=14,
                 fontweight='bold')
    image = cv2.imread(image_path)
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_RECT, kernel_size_3_3)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_RECT, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(3,3)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_CROSS, kernel_size_3_3)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_CROSS, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_ELLIPSE', kernel_size='(3,3)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_ELLIPSE, kernel_size_3_3)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_ELLIPSE', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_ELLIPSE, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)
    plt.show()


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
