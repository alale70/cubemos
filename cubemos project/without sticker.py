#!/usr/bin/env python3
import util as cm
import cv2
import argparse
import os
import platform
from skeletontracker import skeletontracker

import pickle
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Perform keypoing estimation on an image")
parser.add_argument(
    "-c",
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Minimum confidence (0-1) of displayed joints",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Increase output verbosity by enabling backend logging",
)

parser.add_argument(
    "-o",
    "--output_image",
    default="out.avi",
    type=str,
    help="filename of the output image",
)

parser.add_argument(
    "-I",
    "--image",
    type=str,
    default="stream_video1.avi",
    help="filename of the input image",
)

# parser.add_argument("image", metavar="I", type=str, help="filename of the input image")

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

def track_skeletons(self, color_image):
    # perform inference and update the tracking id
    skeletons = self.__api.estimate_keypoints(color_image, 256)
    try:
        skeletons = self.__api.update_tracking(color_image, self.__tracker, skeletons, False)
    except Exception as ex:
        print("Exception occured while updating tracking IDs: \"{}\"".format(ex))

    return skeletons

def get_valid_keypoints(keypoint_ids, skeleton, confs, confidence_threshold):
    keypoints = [
        (tuple(map(int, skeleton[i])), tuple(map(int, skeleton[v])))
        for (i, v) in keypoint_ids
        if confs[i] >= confidence_threshold
           and confs[v] >= confidence_threshold
    ]
    valid_keypoints = [
        keypoint
        for keypoint in keypoints
        if keypoint[0][0] >= 0 and keypoint[0][1] >= 0 and keypoint[1][0] >= 0 and keypoint[1][1] >= 0
    ]
    return valid_keypoints


def render_result(frame_info, img, confidence_threshold):
    points, skeletons = frame_info
    if len(skeletons) == 0:
        return
    skeleton = [x[:2] for x in skeletons]
    confs = [x[2] for x in skeletons]
    skeleton_color = (100, 254, 213)
    keypoints = get_valid_keypoints(keypoint_ids, skeleton, confs, confidence_threshold)
    for keypoint in keypoints:
        cv2.line(img, keypoint[0], keypoint[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA)


def render_point_cordinates(frame_info, img):
    thickness = 1
    text_color = (255, 255, 255)
    points, skeletons = frame_info


    for p in points:
        my_x = load_var("my_x.pckl")
        my_y = load_var("my_y.pckl")
        x, y = p
        point_3d = points[p]

        cv2.putText(img,
                    str(point_3d),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.4,
                    text_color,
                    thickness,
                    )


# Main content begins
if __name__ == "__main__":
    try:
        # Parse command line arguments and check the essentials
        args = parser.parse_args()

        # Get the skeleton tracking object
        skeletrack = skeletontracker()
        joint_confidence = 0.2

        # create window for initialisation
        window_name = "cubemos skeleton tracking with webcam as input source"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap = cv2.VideoCapture(args.image)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outpy.avi', fourcc, 20, (frame_width, frame_height))

        frame_list = load_var("point_cordinate.pckl")
        frame_idx = 0
        points, skeletons = frame_list[frame_idx]
        # Creating Dataset
        X = []
        Y = []
        Z = []
        # perform inference and update the tracking id
        ret, frame = cap.read()
        for i in range(len(skeletons)):
            (x, y, z) = skeletons[i]
            X.append(x)
            Y.append(y)
            Z.append(z)


        # frame *= 0
        def click_event(event, x, y, flags, params):
            # checking for left mouse clicks
            my_x = []
            my_y = []

            if event == cv2.EVENT_LBUTTONDOWN:
                # displaying the coordinates
                # on the Shell
                print(x, ' ', y)
                my_x.append(x)
                my_y.append(y)

                # displaying the coordinates
                # on the image window
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(x) + ',' +
                            str(y), (x, y), font,
                            1, (255, 0, 0), 2)
                cv2.imshow('frame', frame)
                save_var("my_x.pckl", my_x)
                save_var("my_y.pckl", my_y)
            # checking for right mouse clicks
            if event == cv2.EVENT_RBUTTONDOWN:
                # displaying the coordinates
                # on the Shell
                print(x, ' ', y)
                my_x.append(x)
                my_y.append(y)  # displaying the coordinates
                # on the image window
                font = cv2.FONT_HERSHEY_SIMPLEX
                b = frame[y, x, 0]
                g = frame[y, x, 1]
                r = frame[y, x, 2]
                cv2.putText(frame, str(b) + ',' +
                            str(g) + ',' + str(r),
                            (x, y), font, 1,
                            (255, 255, 0), 2)
                cv2.imshow('frame', frame)
                save_var("my_x.pckl", my_x)
                save_var("my_y.pckl", my_y)


        # render_result(frame_list[frame_idx], frame, confidence_threshold=joint_confidence)

        # plt.imshow(frame)
        # plt.show()
        cv2.imshow('frame', frame)
        h = cv2.setMouseCallback('frame', click_event)
        render_point_cordinates(frame_list[frame_idx], frame)
        # cv2.imshow('frame', frame)
        #
        # cv2.setMouseCallback('frame', click_event)

        # write to the output
        # out.write(frame)
        # show the result on on opencv window
        # cv2.imshow(window_name, frame)
        # cv2.waitKey(0)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()
    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))

# Main content ends
