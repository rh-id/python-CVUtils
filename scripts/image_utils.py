import os
import pathlib

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.filters import (threshold_multiotsu, threshold_niblack, threshold_otsu, threshold_sauvola,
                             threshold_triangle, threshold_yen)


def feature_matching_bf(image_path_query, image_path_scene):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(1, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    image_query = cv2.imread(image_path_query)
    image_scene = cv2.imread(image_path_scene)

    # Create the dimensions of the figure and set title:
    fig = plt.figure(figsize=(14, 5))
    plt.suptitle("Feature matching and homography computation for object recognition", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Initiate ORB detector:
    orb = cv2.ORB_create()

    # Detect the keypoints and compute the descriptors with ORB:
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

    # Create BFMatcher object
    # First parameter sets the distance measurement (by default it is cv2.NORM_L2)
    # The second parameter crossCheck (which is False by default) can be set to True in order to return only
    # consistent pairs in the matching process (the two features in both sets should match each other)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors:
    bf_matches = bf_matcher.match(descriptors_1, descriptors_2)
    print("bf_matches: '{}'".format(len(bf_matches)))
    # Sort the matches in the order of their distance:
    bf_matches = sorted(bf_matches, key=lambda x: x.distance)
    best_matches = bf_matches[:40]
    print("best_matches: '{}'".format(len(best_matches)))

    # Extract the matched keypoints:
    pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

    # Find homography matrix:
    M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

    # Get the corner coordinates of the 'query' image:
    h, w = image_query.shape[:2]
    pts_corners_src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Perform perspective transform using the previously calculated matrix and the corners of the 'query' image
    # to get the corners of the 'detected' object in the 'scene' image:
    pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)

    # Draw corners of the detected object:
    img_obj = cv2.polylines(image_scene, [np.int32(pts_corners_dst)], True, (0, 255, 255), 10)

    # Draw matches:
    img_matching = cv2.drawMatches(image_query, keypoints_1, img_obj, keypoints_2, best_matches, None,
                                   matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

    # Plot the images:
    show_img_with_matplotlib(image_query, "img query", 1)
    show_img_with_matplotlib(img_obj, "detected object", 2)
    show_img_with_matplotlib(img_matching, "feature matching", 3)

    # Show the Figure:
    plt.show()


def contours_hu_moments(image_path, export_dir=None):
    def centroid(moments):
        x_centroid = round(moments['m10'] / moments['m00'])
        y_centroid = round(moments['m01'] / moments['m00'])
        return x_centroid, y_centroid

    def draw_contour_outline(img, cnts, color, thickness=1):
        for cnt in cnts:
            cv2.drawContours(img, [cnt], 0, color, thickness)

    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(1, 1, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(12, 5))
    plt.suptitle("Hu moments", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh, True)
    print("moments: '{}'".format(M))

    x, y = centroid(M)

    HuM = cv2.HuMoments(M)
    print("Hu moments: '{}'".format(HuM))

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    M2 = cv2.moments(contours[0])
    print("moments: '{}'".format(M2))

    x2, y2 = centroid(M2)

    HuM2 = cv2.HuMoments(M2)
    print("Hu moments: '{}'".format(HuM2))

    draw_contour_outline(image, contours, (255, 0, 0), 10)

    cv2.circle(image, (x, y), 25, (255, 0, 0), -1)
    cv2.circle(image, (x2, y2), 25, (0, 255, 0), -1)
    print("('x','y'): ('{}','{}')".format(x, y))
    print("('x2','y2'): ('{}','{}')".format(x2, y2))

    show_img_with_matplotlib(image, "detected contour and centroid", 1)

    # Show the Figure:
    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'image.jpg'), image)


def contours_approximation(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Contours approximation method", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

    image_approx_none = image.copy()
    image_approx_simple = image.copy()
    image_approx_tc89_l1 = image.copy()
    image_approx_tc89_kcos = image.copy()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    contours4, hierarchy4 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    cv2.drawContours(image=image_approx_none, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(image=image_approx_simple, contours=contours2, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(image=image_approx_tc89_l1, contours=contours3, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(image=image_approx_tc89_kcos, contours=contours4, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    thresh_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(thresh_img, "threshold = 100", 2)
    show_img_with_matplotlib(image_approx_none, "contours (APPROX_NONE)", 3)
    show_img_with_matplotlib(image_approx_simple, "contours (CHAIN_APPROX_SIMPLE)", 4)
    show_img_with_matplotlib(image_approx_tc89_l1, "contours (APPROX_TC89_L1)", 5)
    show_img_with_matplotlib(image_approx_tc89_kcos, "contours (APPROX_TC89_KCOS)", 6)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'threshold_100.jpg'), thresh_img)
        cv2.imwrite(os.path.join(export_dir, 'approx_none.jpg'), image_approx_none)
        cv2.imwrite(os.path.join(export_dir, 'chain_approx_simple.jpg'), image_approx_simple)
        cv2.imwrite(os.path.join(export_dir, 'approx_tc89_l1.jpg'), image_approx_tc89_l1)
        cv2.imwrite(os.path.join(export_dir, 'approx_tc89_kcos.jpg'), image_approx_tc89_kcos)


def threshold_image_scikit_image(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos, cmap=None, to_rgb=True):
        if to_rgb:
            img_RGB = color_img[:, :, ::-1]
        else:
            img_RGB = color_img

        plt.subplot(3, 3, pos)
        plt.imshow(img_RGB, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Thresholding scikit-image", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_multiotsu = threshold_multiotsu(gray_image)
    multi_otsu = np.digitize(gray_image, bins=thresh_multiotsu)

    thresh_otsu = threshold_otsu(gray_image)
    binary_otsu = gray_image > thresh_otsu
    binary_otsu = img_as_ubyte(binary_otsu)

    thresh_niblack = threshold_niblack(gray_image, window_size=25, k=0.8)
    binary_niblack = gray_image > thresh_niblack
    binary_niblack = img_as_ubyte(binary_niblack)

    thresh_sauvola = threshold_sauvola(gray_image, window_size=25)
    binary_sauvola = gray_image > thresh_sauvola
    binary_sauvola = img_as_ubyte(binary_sauvola)

    thresh_triangle = threshold_triangle(gray_image)
    binary_triangle = gray_image > thresh_triangle
    binary_triangle = img_as_ubyte(binary_triangle)

    thresh_yen = threshold_yen(gray_image)
    binary_yen = gray_image > thresh_yen
    binary_yen = img_as_ubyte(binary_yen)

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    binary_otsu = cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR)
    binary_triangle = cv2.cvtColor(binary_triangle, cv2.COLOR_GRAY2BGR)
    binary_niblack = cv2.cvtColor(binary_niblack, cv2.COLOR_GRAY2BGR)
    binary_sauvola = cv2.cvtColor(binary_sauvola, cv2.COLOR_GRAY2BGR)
    binary_yen = cv2.cvtColor(binary_yen, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(gray_image, "gray img", 2)
    show_img_with_matplotlib(multi_otsu, "Multiotsu", 3, cmap='jet', to_rgb=False)
    show_img_with_matplotlib(binary_otsu, "Otsu", 4)
    show_img_with_matplotlib(binary_triangle, "Triangle",
                             5)
    show_img_with_matplotlib(binary_niblack, "Niblack",
                             6)
    show_img_with_matplotlib(binary_sauvola, "Sauvola",
                             7)
    show_img_with_matplotlib(binary_yen, "Yen",
                             8)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_image.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'otsu.jpg'), binary_otsu)
        cv2.imwrite(os.path.join(export_dir, 'triangle.jpg'), binary_triangle)
        cv2.imwrite(os.path.join(export_dir, 'niblack.jpg'), binary_niblack)
        cv2.imwrite(os.path.join(export_dir, 'sauvola.jpg'), binary_sauvola)
        cv2.imwrite(os.path.join(export_dir, 'yen.jpg'), binary_yen)


def threshold_image_triangle_filter_noise(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, title, pos, color, otsu=-1):
        plt.subplot(2, 3, pos)
        plt.title(title)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])
        plt.axvline(x=otsu, color='m', linestyle='--')
        plt.plot(hist, color=color)

    fig = plt.figure(figsize=(11, 10))
    plt.suptitle("Triangle Binarization with Gaussian filter", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)
    hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])
    ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
    th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(gray_image, "gray img", 2)
    show_hist_with_matplotlib_gray(hist, "gray img", 3, 'm', ret1)
    show_img_with_matplotlib(th1,
                             "Triangle binarization (before Gaussian filter)", 4)
    show_hist_with_matplotlib_gray(hist2, "gray img (after Gaussian filter)", 5, 'm', ret2)
    show_img_with_matplotlib(th2,
                             "Triangle binarization (after Gaussian filter)", 6)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_image.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'triangle_before_gaussian.jpg'), th1)
        cv2.imwrite(os.path.join(export_dir, 'triangle_after_gaussian.jpg'), th2)


def threshold_image_otsu_filter_noise(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
        plt.subplot(2, 3, pos)
        plt.title(title)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])
        plt.axvline(x=t, color='m', linestyle='--')
        plt.plot(hist, color=color)

    fig = plt.figure(figsize=(11, 10))
    plt.suptitle("Otsu's Binarization Filter Noise", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)

    hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])

    ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
    th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(image, "image with noise", 1)
    show_img_with_matplotlib(gray_image, "gray img", 2)
    show_hist_with_matplotlib_gray(hist, "gray img", 3, 'm', ret1)
    show_img_with_matplotlib(th1,
                             "Otsu's binarization (before Gaussian filter)", 4)
    show_hist_with_matplotlib_gray(hist2, "gray img (Gaussian filter)", 5, 'm', ret2)
    show_img_with_matplotlib(th2,
                             "Otsu's binarization (after Gaussian filter)", 6)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_image.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'otsu_before_gaussian.jpg'), th1)
        cv2.imwrite(os.path.join(export_dir, 'otsu_after_gaussian.jpg'), th2)


def threshold_image_otsu(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(2, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
        plt.subplot(2, 3, pos)
        plt.title(title)
        plt.xlabel("bins")
        plt.ylabel("number of pixels")
        plt.xlim([0, 256])
        plt.axvline(x=t, color='m', linestyle='--')
        plt.plot(hist, color=color)

    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

    th1_bgr = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)

    hist_th1 = cv2.calcHist([th1_bgr], [0], None, [256], [0, 256])

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(gray_image, "gray img", 2)
    show_hist_with_matplotlib_gray(hist, "gray img", 3, 'm', ret1)
    show_img_with_matplotlib(th1_bgr, "Otsu's binarization", 4)
    show_hist_with_matplotlib_gray(hist_th1, "Otsu's binarization", 5, 'm')

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_image.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'otsu.jpg'), th1_bgr)


def threshold_image_bgr(image_path, export_dir=None):
    def show_img_with_matplotlib(color_img, title, pos):
        img_RGB = color_img[:, :, ::-1]

        plt.subplot(1, 3, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(12, 4))
    plt.suptitle("Thresholding BGR", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    image = cv2.imread(image_path)

    show_img_with_matplotlib(image, "image", 1)

    ret1, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    (b, g, r) = cv2.split(image)
    ret2, thresh2 = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
    ret4, thresh4 = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
    bgr_thresh = cv2.merge((thresh2, thresh3, thresh4))

    show_img_with_matplotlib(thresh1, "threshold (120) BGR", 2)
    show_img_with_matplotlib(bgr_thresh, "threshold (120) each channel and merge", 3)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'threshold_120_BGR.jpg'), thresh1)
        cv2.imwrite(os.path.join(export_dir, 'threshold_120_each_channel_and_merge.jpg'), bgr_thresh)


def threshold_image_adaptive_filter_noise(image_path, export_dir=None):
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

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
    thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR)
    thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(gray_image, "gray img", 1)
    show_img_with_matplotlib(thresh1, "THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(thresh2, "THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(thresh3, "GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(thresh4, "GAUSSIAN_C, blockSize=31, C=3", 6)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_img.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'thresh_mean_c_b11_c2.jpg'), thresh1)
        cv2.imwrite(os.path.join(export_dir, 'thresh_mean_c_b31_c3.jpg'), thresh2)
        cv2.imwrite(os.path.join(export_dir, 'gaussian_c_b11_c2.jpg'), thresh3)
        cv2.imwrite(os.path.join(export_dir, 'gaussian_c_b31_c3.jpg'), thresh4)


def threshold_image_adaptive(image_path, export_dir=None):
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

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
    thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR)
    thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(gray_image, "gray img", 1)
    show_img_with_matplotlib(thresh1, "THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(thresh2, "THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(thresh3, "GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(thresh4, "GAUSSIAN_C, blockSize=31, C=3", 6)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_img.jpg'), gray_image)
        cv2.imwrite(os.path.join(export_dir, 'thresh_mean_c_b11_c2.jpg'), thresh1)
        cv2.imwrite(os.path.join(export_dir, 'thresh_mean_c_b31_c3.jpg'), thresh2)
        cv2.imwrite(os.path.join(export_dir, 'gaussian_c_b11_c2.jpg'), thresh3)
        cv2.imwrite(os.path.join(export_dir, 'gaussian_c_b31_c3.jpg'), thresh4)


def threshold_image_binary(image_path, export_dir=None):
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

    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
    thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR)
    thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR)
    thresh5 = cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR)
    thresh6 = cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR)
    thresh7 = cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR)
    thresh8 = cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(thresh1, "threshold = 60", 2)
    show_img_with_matplotlib(thresh2, "threshold = 70", 3)
    show_img_with_matplotlib(thresh3, "threshold = 80", 4)
    show_img_with_matplotlib(thresh4, "threshold = 90", 5)
    show_img_with_matplotlib(thresh5, "threshold = 100", 6)
    show_img_with_matplotlib(thresh6, "threshold = 110", 7)
    show_img_with_matplotlib(thresh7, "threshold = 120", 8)
    show_img_with_matplotlib(thresh8, "threshold = 130", 9)

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'threshold_60.jpg'), thresh1)
        cv2.imwrite(os.path.join(export_dir, 'threshold_70.jpg'), thresh2)
        cv2.imwrite(os.path.join(export_dir, 'threshold_80.jpg'), thresh3)
        cv2.imwrite(os.path.join(export_dir, 'threshold_90.jpg'), thresh4)
        cv2.imwrite(os.path.join(export_dir, 'threshold_100.jpg'), thresh5)
        cv2.imwrite(os.path.join(export_dir, 'threshold_110.jpg'), thresh6)
        cv2.imwrite(os.path.join(export_dir, 'threshold_120.jpg'), thresh7)
        cv2.imwrite(os.path.join(export_dir, 'threshold_130.jpg'), thresh8)


def histogram_image_color_equalize_hsv(image_path, export_dir=None):
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

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'image_lighter.jpg'), added_image)
        cv2.imwrite(os.path.join(export_dir, 'image_darker.jpg'), subtracted_image)
        cv2.imwrite(os.path.join(export_dir, 'image_equalized.jpg'), image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_lighter_equalized.jpg'), added_image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_darker_equalized.jpg'), subtracted_image_eq)


def histogram_image_color_equalize(image_path, export_dir=None):
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

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'image_lighter.jpg'), added_image)
        cv2.imwrite(os.path.join(export_dir, 'image_darker.jpg'), subtracted_image)
        cv2.imwrite(os.path.join(export_dir, 'image_equalize.jpg'), image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_lighter_equalize.jpg'), added_image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_darker_equalize.jpg'), subtracted_image_eq)


def histogram_image_color(image_path, export_dir=None):
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

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'image_lighter.jpg'), added_image)
        cv2.imwrite(os.path.join(export_dir, 'image_darker.jpg'), subtracted_image)


def histogram_image_gray_mask(image_path, export_dir=None):
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
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(masked_img, "masked gray image", 3)
    show_hist_with_matplotlib_gray(hist_mask, 4, 'm')

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'masked_gray_image.jpg'), masked_img)


def histogram_image_gray_equalize(image_path, export_dir=None):
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

    added_image = cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR)
    subtracted_image = cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
    show_hist_with_matplotlib_gray(hist, 2, 'm')
    show_img_with_matplotlib(added_image, "gray lighter", 5)
    show_hist_with_matplotlib_gray(hist_added_image, 6, 'm')
    show_img_with_matplotlib(subtracted_image, "gray darker", 9)
    show_hist_with_matplotlib_gray(hist_subtracted_image, 10, 'm')

    gray_image_eq = cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR)
    added_image_eq = cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR)
    subtracted_image_eq = cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(gray_image_eq, "grayscale equalized", 3)
    show_hist_with_matplotlib_gray(hist_eq, 4, 'm')
    show_img_with_matplotlib(added_image_eq, "gray lighter equalized", 7)
    show_hist_with_matplotlib_gray(hist_eq_added_image, 8, 'm')
    show_img_with_matplotlib(subtracted_image_eq, "gray darker equalized", 11)
    show_hist_with_matplotlib_gray(hist_eq_subtracted_image, 12, 'm')

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'image_lighter.jpg'), added_image)
        cv2.imwrite(os.path.join(export_dir, 'image_darker.jpg'), subtracted_image)
        cv2.imwrite(os.path.join(export_dir, 'image_equalize.jpg'), gray_image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_lighter_equalize.jpg'), added_image_eq)
        cv2.imwrite(os.path.join(export_dir, 'image_darker_equalize.jpg'), subtracted_image_eq)


def histogram_image_gray(image_path, export_dir=None):
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

    added_image = cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR)
    subtracted_image = cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR)
    show_img_with_matplotlib(added_image, "gray lighter", 2)
    show_hist_with_matplotlib_gray(hist_added_image, 5, 'm')
    show_img_with_matplotlib(subtracted_image, "gray darker", 3)
    show_hist_with_matplotlib_gray(hist_subtracted_image, 6, 'm')

    plt.show()

    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'gray_lighter.jpg'), added_image)
        cv2.imwrite(os.path.join(export_dir, 'gray_darker.jpg'), subtracted_image)


def analyze_image_kernel(image_path, export_dir=None):
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

    # Export if applicable
    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(export_dir, 'edge_detection_1.jpg'), edge_image_1)
        cv2.imwrite(os.path.join(export_dir, 'edge_detection_2.jpg'), edge_image_2)
        cv2.imwrite(os.path.join(export_dir, 'edge_detection_3.jpg'), edge_image_3)
        cv2.imwrite(os.path.join(export_dir, 'sharpen_image.jpg'), sharpen_image)
        cv2.imwrite(os.path.join(export_dir, 'unsharp_masking.jpg'), unsharp_masking_image)
        cv2.imwrite(os.path.join(export_dir, 'blur_image.jpg'), blur_image)
        cv2.imwrite(os.path.join(export_dir, 'gaussian_blur_image.jpg'), gaussian_blur_image)
        cv2.imwrite(os.path.join(export_dir, 'emboss_image.jpg'), emboss_image)
        cv2.imwrite(os.path.join(export_dir, 'sobel_x_image.jpg'), sobel_x_image)
        cv2.imwrite(os.path.join(export_dir, 'sobel_y_image.jpg'), sobel_y_image)
        cv2.imwrite(os.path.join(export_dir, 'outline_image.jpg'), outline_image)


def analyze_image_skin_segment(image_path, export_dir=None):
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

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

            cv2.imwrite(os.path.join(export_dir, k + '.jpg'), bgr)
    plt.show()


def analyze_image_morph(image_path, export_dir=None):
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

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'rect_3x3_' + k + '.jpg'), image_morph)

    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_RECT, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'rect_5x5_' + k + '.jpg'), image_morph)

    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(3,3)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_CROSS, kernel_size_3_3)
        show_with_matplotlib(image_morph, k, i + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'cross_3x3_' + k + '.jpg'), image_morph)

    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_CROSS, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'cross_5x5_' + k + '.jpg'), image_morph)

    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_ELLIPSE', kernel_size='(3,3)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_ELLIPSE, kernel_size_3_3)
        show_with_matplotlib(image_morph, k, i + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'ellipse_3x3_' + k + '.jpg'), image_morph)

    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_ELLIPSE', kernel_size='(5,5)'", fontsize=14,
                 fontweight='bold')
    show_with_matplotlib(image, 'Input Image', 1)
    for i, (k, v) in enumerate(morphological_operations.items()):
        image_morph = v(image, cv2.MORPH_ELLIPSE, kernel_size_5_5)
        show_with_matplotlib(image_morph, k, i + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, 'ellipse_5x5_' + k + '.jpg'), image_morph)
    plt.show()


def analyze_image_color_map(image_path, export_dir=None):
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
        image = cv2.applyColorMap(gray_img, idx)
        show_with_matplotlib(image, val, idx + 2)

        # Export if applicable
        if export_dir is not None:
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(export_dir, val + '.jpg'), image)

    plt.show()


# Analyze image with color map
def analyze_image_color_space(image_path, export_dir=None):
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

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(gray_image, "gray image", 1 + 6)

    bgr_b = cv2.cvtColor(bgr_b, cv2.COLOR_GRAY2BGR)
    bgr_g = cv2.cvtColor(bgr_g, cv2.COLOR_GRAY2BGR)
    bgr_r = cv2.cvtColor(bgr_r, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(bgr_b, "BGR - B comp", 2)
    show_with_matplotlib(bgr_g, "BGR - G comp", 2 + 6)
    show_with_matplotlib(bgr_r, "BGR - R comp", 2 + 6 * 2)

    hsv_h = cv2.cvtColor(hsv_h, cv2.COLOR_GRAY2BGR)
    hsv_s = cv2.cvtColor(hsv_s, cv2.COLOR_GRAY2BGR)
    hsv_v = cv2.cvtColor(hsv_v, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(hsv_h, "HSV - H comp", 3)
    show_with_matplotlib(hsv_s, "HSV - S comp", 3 + 6)
    show_with_matplotlib(hsv_v, "HSV - V comp", 3 + 6 * 2)

    hls_h = cv2.cvtColor(hls_h, cv2.COLOR_GRAY2BGR)
    hls_l = cv2.cvtColor(hls_l, cv2.COLOR_GRAY2BGR)
    hls_s = cv2.cvtColor(hls_s, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(hls_h, "HLS - H comp", 4)
    show_with_matplotlib(hls_l, "HLS - L comp", 4 + 6)
    show_with_matplotlib(hls_s, "HLS - S comp", 4 + 6 * 2)

    ycrcb_y = cv2.cvtColor(ycrcb_y, cv2.COLOR_GRAY2BGR)
    ycrcb_cr = cv2.cvtColor(ycrcb_cr, cv2.COLOR_GRAY2BGR)
    ycrcb_cb = cv2.cvtColor(ycrcb_cb, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(ycrcb_y, "YCrCb - Y comp", 5)
    show_with_matplotlib(ycrcb_cr, "YCrCb - Cr comp", 5 + 6)
    show_with_matplotlib(ycrcb_cb, "YCrCb - Cb comp", 5 + 6 * 2)

    lab_l = cv2.cvtColor(lab_l, cv2.COLOR_GRAY2BGR)
    lab_a = cv2.cvtColor(lab_a, cv2.COLOR_GRAY2BGR)
    lab_b = cv2.cvtColor(lab_b, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(lab_l, "L*a*b - L comp", 6)
    show_with_matplotlib(lab_a, "L*a*b - a comp", 6 + 6)
    show_with_matplotlib(lab_b, "L*a*b - b comp", 6 + 6 * 2)

    # Export if applicable
    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(export_dir, 'gray_image.jpg'), gray_image)

        cv2.imwrite(os.path.join(export_dir, 'bgr_b.jpg'), bgr_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_g.jpg'), bgr_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_r.jpg'), bgr_r)

        cv2.imwrite(os.path.join(export_dir, 'hsv_h.jpg'), hsv_h)
        cv2.imwrite(os.path.join(export_dir, 'hsv_s.jpg'), hsv_s)
        cv2.imwrite(os.path.join(export_dir, 'hsv_v.jpg'), hsv_v)

        cv2.imwrite(os.path.join(export_dir, 'hls_h.jpg'), hls_h)
        cv2.imwrite(os.path.join(export_dir, 'hls_l.jpg'), hls_l)
        cv2.imwrite(os.path.join(export_dir, 'hls_s.jpg'), hls_s)

        cv2.imwrite(os.path.join(export_dir, 'ycrcb_y.jpg'), ycrcb_y)
        cv2.imwrite(os.path.join(export_dir, 'ycrcb_cr.jpg'), ycrcb_cr)
        cv2.imwrite(os.path.join(export_dir, 'ycrcb_cb.jpg'), ycrcb_cb)

        cv2.imwrite(os.path.join(export_dir, 'lab_l.jpg'), lab_l)
        cv2.imwrite(os.path.join(export_dir, 'lab_a.jpg'), lab_a)
        cv2.imwrite(os.path.join(export_dir, 'lab_b.jpg'), lab_b)

    plt.show()


# Analyze image with color split
def analyze_image_color_split(image_path, export_dir=None):
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

    bgr_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    bgr_g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr_r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(bgr_b, "BGR - (B)", 2)
    show_with_matplotlib(bgr_g, "BGR - (G)", 2 + 6)
    show_with_matplotlib(bgr_r, "BGR - (R)", 2 + 6 * 2)

    image_copy = cv2.merge((b, g, r))

    show_with_matplotlib(image_copy, "BGR - image (copy)", 1 + 6)

    bgr_without_b = image.copy()
    bgr_without_b[:, :, 0] = 0
    bgr_without_g = image.copy()
    bgr_without_g[:, :, 1] = 0
    bgr_without_r = image.copy()
    bgr_without_r[:, :, 2] = 0

    show_with_matplotlib(bgr_without_b, "BGR without B", 3)
    show_with_matplotlib(bgr_without_g, "BGR without G", 3 + 6)
    show_with_matplotlib(bgr_without_r, "BGR without R", 3 + 6 * 2)

    (b, g, r) = cv2.split(bgr_without_b)
    bgr_without_b_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    bgr_without_b_g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr_without_b_r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    show_with_matplotlib(bgr_without_b_b, "BGR without B (B)", 4)
    show_with_matplotlib(bgr_without_b_g, "BGR without B (G)", 4 + 6)
    show_with_matplotlib(bgr_without_b_r, "BGR without B (R)", 4 + 6 * 2)

    (b, g, r) = cv2.split(bgr_without_g)
    bgr_without_g_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    bgr_without_g_g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr_without_g_r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    show_with_matplotlib(bgr_without_g_b, "BGR without G (B)", 5)
    show_with_matplotlib(bgr_without_g_g, "BGR without G (G)", 5 + 6)
    show_with_matplotlib(bgr_without_g_r, "BGR without G (R)", 5 + 6 * 2)

    (b, g, r) = cv2.split(bgr_without_r)
    bgr_without_r_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    bgr_without_r_g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr_without_r_r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    show_with_matplotlib(bgr_without_r_b, "BGR without R (B)", 6)
    show_with_matplotlib(bgr_without_r_g, "BGR without R (G)", 6 + 6)
    show_with_matplotlib(bgr_without_r_r, "BGR without R (R)", 6 + 6 * 2)

    # Export if applicable
    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(export_dir, 'bgr_b.jpg'), bgr_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_g.jpg'), bgr_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_r.jpg'), bgr_r)

        cv2.imwrite(os.path.join(export_dir, 'bgr_without_b.jpg'), bgr_without_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_g.jpg'), bgr_without_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_r.jpg'), bgr_without_r)

        cv2.imwrite(os.path.join(export_dir, 'bgr_without_b_b.jpg'), bgr_without_b_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_b_g.jpg'), bgr_without_b_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_b_r.jpg'), bgr_without_b_r)

        cv2.imwrite(os.path.join(export_dir, 'bgr_without_g_b.jpg'), bgr_without_g_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_g_g.jpg'), bgr_without_g_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_g_r.jpg'), bgr_without_g_r)

        cv2.imwrite(os.path.join(export_dir, 'bgr_without_r_b.jpg'), bgr_without_r_b)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_r_g.jpg'), bgr_without_r_g)
        cv2.imwrite(os.path.join(export_dir, 'bgr_without_r_r.jpg'), bgr_without_r_r)

    plt.show()


# Analyze image with multiple sharpen tech
def analyze_image_sharpen(image_path, export_dir=None):
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

    # Export if applicable
    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(export_dir, 'sharp_1.jpg'), sharp_image_1)
        cv2.imwrite(os.path.join(export_dir, 'sharp_2.jpg'), sharp_image_2)
        cv2.imwrite(os.path.join(export_dir, 'sharp_3.jpg'), sharp_image_3)
        cv2.imwrite(os.path.join(export_dir, 'sharp_4.jpg'), sharp_image_4)
        cv2.imwrite(os.path.join(export_dir, 'sharp_5.jpg'), sharp_image_5)

    plt.show()


# Analyze image with multiple smooth tech
def analyze_image_smooth(image_path, export_dir=None):
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

    # Export if applicable
    if export_dir is not None:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_f2D_5x5.jpg'), smooth_image_f2D_5_5)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_f2D_10x10.jpg'), smooth_image_f2D_10_10)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_blur.jpg'), smooth_image_b)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_box_filter.jpg'), smooth_image_bfi)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_gaussian_blur.jpg'), smooth_image_gb)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_median_blur.jpg'), smooth_image_mb)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_bilateral_filter_small.jpg'), smooth_image_bf)
        cv2.imwrite(os.path.join(export_dir, 'smooth_image_bilateral_filter_big.jpg'), smooth_image_bf_2)

    plt.show()
