import cv2
import numpy as np
# Problem in finding the correct length of sign 

cap = cv2.VideoCapture(0)
wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
x,fr = cap.read()

print(fr)

last_f = None 
sign_frames = []
while True:
    x,frame = cap.read()
    if x:
        s =32
        frame = cv2.resize(frame,(s,s),cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if type(last_f) == type(None) and x:
            last_f = np.array(frame,dtype="uint8")

        if type(last_f) != type(None): #      
            diffed = cv2.absdiff(frame,last_f) 
            diffed[diffed<30] =0 
            count = np.count_nonzero(diffed)
            print(count)
            if count  > 10:
            	cv2.imshow('frame',frame)	
            last_f = np.array(frame,dtype="uint8")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# while True:
#     x,frame = cap.read()
#     if x:
#         s =64 
#         frame = cv2.resize(frame,(s,s),cv2.INTER_CUBIC)

#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         lower_blue = np.array([0, 0, 120])
#         upper_blue = np.array([180, 38, 255])
#         mask = cv2.inRange(hsv, lower_blue, upper_blue)
#         result = cv2.bitwise_and(frame, frame, mask=mask)
#         b, g, r = cv2.split(result)  
#         filter = g.copy()
#         ret,mask = cv2.threshold(filter,10,255, 1)
#         frame[ mask == 0] = 255
#         cv2.imshow('frame',frame)	
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

      

#         continue


#         ret,mask = cv2.threshold(filter,10,255, 1) 
#         frame[ mask == 0] = 255
#         depth_back = depth_array.copy()
#         depth_diff = abs(depth_array - depth_back)
#         median = np.median(depth_array[depth_array > 0])                
#         mask = depth_array.copy()
#         mask[mask > median+0.07 ] = 0
#         mask[mask < median-0.07 ] = 0
#         mask[mask > 0] = 1