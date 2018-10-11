# coding=utf-8
from __future__ import print_function, absolute_import, division

import cv2
import numpy as np

cap = cv2.VideoCapture("/home/jwwangchn/Documents/Research/Projects/2018-Bottle-Detection/07-Videos/5-多目标检测-RealSense视角.avi")

count = 1
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imwrite("./data/{}.jpg".format(count), frame)
    count = count + 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()