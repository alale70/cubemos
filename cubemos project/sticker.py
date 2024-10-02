import cv2
import numpy as np

img = cv2.imread('ax_Color.png', cv2.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 58
#get threshold image
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
# img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img, contours, -1, (0,255,0), 3)
#save image
cv2.imwrite('contours.png', img)
