import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('testimage1.png')   # you can read in images with opencv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_color1 = np.asarray([0, 10, 60])   # white!
hsv_color2 = np.asarray([20, 150, 255])   # yellow! note the order

mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)


mask[mask < 30] = 0
print(mask.shape)

cap = cv2.VideoCapture(0)
x = False 
while not x:
    x,pframe = cap.read()
pframe = cv2.resize(pframe,(16,16),0,0,cv2.INTER_CUBIC)
pframe= cv2.cvtColor(pframe,cv2.COLOR_BGR2HSV)
pframe = cv2.inRange(pframe,hsv_color1,hsv_color2)


while True:

    x,frame = cap.read() 
    frame = cv2.resize(frame,(16,16),0,0,cv2.INTER_CUBIC)
    frame= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame,hsv_color1,hsv_color2)
    print(frame.shape) 
    diff = cv2.absdiff(frame,pframe)
    print(diff.shape)
    diff[diff<80] = 0
    pframe = np.array(frame)
    cv2.imshow('frame',diff)
    cv2.imshow('fram1',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break 
            