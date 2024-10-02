import cv2
import numpy as np

img = cv2.imread('ax_Color.png', cv2.IMREAD_UNCHANGED)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower_blue = np.array([58, 155, 84])
upper_blue = np.array([179, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

Contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# centimeters = pixels * 2.54 / 96

cv2.drawContours(img, Contours, -1, (0, 255, 0), 3)
cv2.imshow("Frame", img)
cv2.imshow("Mask", mask)
cv2.imwrite('jointMask.png', img)
el = Contours[0]
el2 = (el[0] * 2.54)/96

el3 = Contours[5]
el4 = (el3[0] * 2.54)/96
key = cv2.waitKey(100000)
KNOWN_DISTANCE = 150
KNOWN_WIDTH = 5

focalLength = (el2[0][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


z_value = distance_to_camera(KNOWN_WIDTH, focalLength, el4[0][0])
print(z_value)
