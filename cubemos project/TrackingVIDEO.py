#!/usr/bin/env python3
import util as cm
import cv2
import argparse
import os
import platform
from skeletontracker import skeletontracker

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
    default="out.mp4",
    type=str,
    help="filename of the output image",
)

parser.add_argument(
    "-I",
    "--image",
    type=str,
    default="2.mp4",
    help="filename of the input image",
)

#parser.add_argument("image", metavar="I", type=str, help="filename of the input image")


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

        out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        while(cap.isOpened()):
            # perform inference and update the tracking id
            ret, frame = cap.read()

            skeletons = skeletrack.track_skeletons(frame)

            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, frame, joint_confidence)
            cm.render_ids(skeletons, frame)

            # write to the output
            out.write(frame)
            # show the result on on opencv window
            cv2.imshow(window_name, frame)
            # cv2.waitKey(0)
            if cv2.waitKey(1) == 27:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))





# Main content ends
