import cv2
import matplotlib.pyplot as plt
import numpy as np 
import os 
import time 

from icecream import ic
import pickle 
dataset = []
data_loc = "/Volumes/Nantha/Workspace/Datasets/Nanthak Dataset/"

#data_loc = "drive/MyDrive/My Dataset/"
labels = []
processed_videos = []
def plot_frames(fs,n):
    columns = n//(n//5)
    rows = (n//5) 
    fig=plt.figure(figsize=(6, 6))
    print(len(fs))
    for i in range(1,rows*columns+1):
        img = fs[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


processed_videos = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/allvideos.pkl","rb"))
labels = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/alllabels.pkl","rb"))
print(labels[0])
plot_frames(processed_videos[0],40)
exit()
def get_data_nframes(data_loca,n):
    # for folder in os.listdir(data_loc):
    #     if folder[0] != '.':
    seen = []
    prev_tag = None
    for file in os.listdir(data_loca):
        if file[0]==".":
            continue 
        cap = cv2.VideoCapture(data_loca  + "/"+ file)
        
        name = file.split("_")
        if name[0] == 'v':
            tag = name[1].lower()
        else:
            tag = name[0].lower()

        if tag in seen:
            continue
        print(tag)
        w = 128
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fcount == 0:
            continue 
        print("A",fcount)


        for start in range(0,fcount//n):
            cap.set(1,start)
            suc,frame = cap.read()
            if not suc:
                continue

            pframe = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            pframe = cv2.cvtColor(pframe,cv2.COLOR_BGR2GRAY)
        
            suc,frame = cap.read()
            p1frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            p1frame = cv2.cvtColor(p1frame,cv2.COLOR_BGR2GRAY)

            pframea = cv2.absdiff(pframe,p1frame)
            pframea[pframea < 30 ] = 0  #denoising 
            all_frames = []
            print("B",fcount,n)
            for i in range(start + fcount//n,fcount,fcount//n):
                
                cap.set(1,i-1)
                x,frame = cap.read()
                if not x:
                    continue

                frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                x,frame1 = cap.read()
                if not x:
                    continue 

                frame1 = cv2.resize(frame1,(w,w),0,0,cv2.INTER_CUBIC)
                frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                framea = cv2.absdiff(frame,frame1)

                framea[framea < 30] = 0  #denoising

                diff = cv2.absdiff(framea,pframea)
                diff[diff<30] = 0
            
                # count = np.count_nonzero(diff)
                all_frames.append(diff)
                pframea = np.array(framea)

            if len(all_frames) > 8:
            
                while len(all_frames) < n:
                    all_frames.append(all_frames[-1])
                all_frames = all_frames[:n] 
                print("Frames in this video:",len(all_frames))
                labels.append(tag)
                processed_videos.append(all_frames)
        
                seen.append(tag)
            for frm in all_frames:
                cv2.imshow('frame',frm)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'): break 
            if len(all_frames) < n:
                plot_frames(all_frames,n)


n = 10
#get_data_nframes("/Volumes/Nantha/Workspace/Datasets/Nanthak Dataset/",n)


get_data_nframes("/Volumes/Nantha/Workspace/Datasets/Ram Dataset/",n)
processed_videos = np.array(processed_videos)
processed_videos = processed_videos[:,np.newaxis,:,:,:]
(np.array(processed_videos).shape)