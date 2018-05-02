import cv2
import numpy as np
import math
import copy

import os

w = 1200
h = 900

cap = cv2.VideoCapture(0)


cv2.namedWindow('Snapshot')

i = 0
while True:
    ret, img = cap.read()

    cv2.imshow('Snapshot',img)

    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 
        exit()
    elif key & 0xFF == ord('p'):
        cv2.imwrite(str(i)+'.jpg', img)
        i = i + 1