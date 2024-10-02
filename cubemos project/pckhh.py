import util as cm
import cv2
import pyautogui

import time
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker
import pyrealsense2 as rs

import matplotlib.pyplot as plt
import pickle

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17)
]


def pckh(headsizes, pos_pred_src, pos_gt_src, threshold):
    jnt_visible = 18
    uv_err = tuple(map(lambda i, j: i - j, pos_pred_src, pos_gt_src))


    # scale = (headsizes, headsizes)
    scale = headsizes

    scaled_uv_err = tuple(elem_1 // elem_2 for elem_1, elem_2 in zip(uv_err, scale))

    # scaled_uv_err = scaled_uv_err * jnt_visible
    jnt_count = 18
    less_than_threshold = (scaled_uv_err[0] <= threshold) * jnt_visible
    PCKh = 100. * np.sum(less_than_threshold) / jnt_count
    pckh_list.append(PCKh)
    save_var("pckh_list.pckl", pckh_list)




def get_valid_keypoints(keypoint_ids, skeleton, confidence_threshold):
    keypoints = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
           and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_keypoints = [
        keypoint
        for keypoint in keypoints
        if keypoint[0][0] >= 0 and keypoint[0][1] >= 0 and keypoint[1][0] >= 0 and keypoint[1][1] >= 0
    ]
    skeleton_color = (100, 254, 213)
    for keypoint in keypoints:
        cv2.line(
            color_image, keypoint[0], keypoint[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA
        )

    return valid_keypoints


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    for index, skeleton in enumerate(skeletons):
        keypoints = get_valid_keypoints(keypoint_ids, skeleton, confidence_threshold)

    return keypoints


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        my_list.append((x, y))

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_image, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', color_image)
        save_var("my_list.pckl", my_list)
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        my_list.append((x, y))
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = color_image[y, x, 0]
        g = color_image[y, x, 1]
        r = color_image[y, x, 2]
        cv2.putText(color_image, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('frame', color_image)
        save_var("my_list.pckl", my_list)


if __name__ == "__main__":
    try:

        # Convert images to numpy arrays
        color_image = cv2.imread("8.jpg")

        skeletrack = skeletontracker()

        # Perform skeleton tracking
        skeletons = skeletrack.track_skeletons(color_image)

        # KEY = render_result(skeletons, color_image, 0.2)
        for index, skeleton in enumerate(skeletons):
            keypoints = get_valid_keypoints(keypoint_ids, skeleton, 0.2)
        list1 = []
        KEY = get_valid_keypoints(keypoint_ids, skeleton, 0.2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for keypoint in KEY:
            list1.append(keypoint[0])
            list1.append(keypoint[1])

        cubemos_coordinate = list(dict.fromkeys(list1))

        # for i in list2:
        #     cv2.putText(color_image, str(i), i, font, 0.5,
        #                 (255, 255, 0), 1)
        cv2.imshow('image', color_image)
        my_list=[]
        pckh_list = []

        cv2.setMouseCallback('image', click_event)

        # PCKH

        file = open('my_list.pckl', 'rb')
        my_coordinate = pickle.load(file)
        file.close()
        headsizes= tuple(map(lambda i, j: i - j, my_coordinate[16], my_coordinate[17]))

        for i in range(18):
            pos_pred_src = cubemos_coordinate[i]
            pos_gt_src = my_coordinate[i]
            pckh(headsizes,pos_pred_src, pos_gt_src, 0.2)

        file = open('pckh_list.pckl', 'rb')
        my_pckh = pickle.load(file)
        final_pckh=(np.sum(my_pckh))/1800
        file.close()
        print("PCKh of Cubemos is",final_pckh)
        cv2.waitKey(0)

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
