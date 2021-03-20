import os
import numpy as np
import cv2
import math

tags1 = set()
labels2 = []
dataset1 = []
required_files = os.listdir("../dataset/Word/")
count = 0
for filename in required_files:
    splitted = filename.split('_')
    print(filename,':',required_files.index(filename),'/',len(required_files))
    
    tags1.add(splitted[1].lower())
    images = []
    cap = cv2.VideoCapture("../dataset/Word"+"/"+filename)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,128);
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    j = 0
    required_count = 30
    cap.set(1,1)
    suc, prev_frame = cap.read()
    while not suc:
        suc,prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (128,128), 0, 0, cv2.INTER_CUBIC);
    prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
    count+=1

    try:
        while j < framecount:
            i = math.ceil(j)
            j+=1
            cap.set(1,i)
            success, frame = cap.read()
            
            if (type(frame) != type(None) ):
                frame = cv2.resize(frame, (128,128), 0, 0, cv2.INTER_CUBIC);
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(frame,prev_frame)
                non_zero_count = np.count_nonzero(diff)
                if non_zero_count > 15000:
                    images.append(diff)
        images = np.array(images)
        dataset1.append(images)
        labels2.append(splitted[1].lower())
        
        print(splitted[1].lower(), images.shape, count,'/',len(required_files))

    except:
        print("Error with ",splitted[1].lower())
    cap.release()
    cv2.destroyAllWindows()
print(len(dataset1))
