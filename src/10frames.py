import csv
import os
import cv2
import numpy as np
import time 
from icecream import ic



dataset = []
data_loc = "/Volumes/Nantha/Workspace/Datasets/Nanthak Dataset/"
labels = []
processed_videos = []

for folder in os.listdir(data_loc):
    if folder[0] != '.':
        for file in os.listdir(data_loc+folder):
            if file[0] == ".":
                continue

            cap = cv2.VideoCapture(data_loc + folder + "/"+ file)
           
            name = file.split("_")

            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            
            if tag in labels:
                continue
            w =128 
            fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            suc,frame = cap.read()
            if not suc:
                print("Not succeded")
                continue
            print("Succeded")
            pframe = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            pframe = cv2.cvtColor(pframe,cv2.COLOR_BGR2GRAY)

            suc,frame = cap.read()
            p1frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            p1frame = cv2.cvtColor(p1frame,cv2.COLOR_BGR2GRAY)

            pframea = cv2.absdiff(pframe,p1frame)  
 
            all_frames = []

            for i in range(fcount//20, fcount, fcount//20):
                x,frame = cap.read()
                if not x:
                    continue
 
                frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                x,frame1 = cap.read()
                if not x:
                    continue
                frame1 = cv2.resize(frame1,(w,w),0,0,cv2.INTER_CUBIC)
                frame1= cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                framea = cv2.absdiff(frame,frame1) 

                cap.set(1,i+fcount//20-1)
                if not x:
                    print("failure")
                    continue
               
                diff = cv2.absdiff(framea,pframea)
                diff[diff<30] = 0
                count = np.count_nonzero(diff)
                # cv2.imshow('frame',diff)
                # time.sleep(0.1)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break 
                all_frames.append(diff)  
                pframea = np.array(framea) 
                



            if len(all_frames) > 8:
                while len(all_frames) < 10:
                    all_frames.append(all_frames[-1])
                labels.append(tag)
                processed_videos.append(all_frames)
            for frm in all_frames:
                cv2.imshow('frame',frm)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'): break 

            cap.release()
            # for i in range(1,fcount):
            #     x,frame = cap.read()
            #     if not x:
            #         continue
            #     frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            #     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #     diff = cv2.absdiff(frame,pframe)
            #     diff[diff<30] = 0
            #     count = np.count_nonzero(diff)
            #     if count > 10:
            #         # cv2.imshow('frame',diff)
            #         # if cv2.waitKey(1) & 0xFF == ord('q'): break 
            #         all_frames.append(diff)
            #     pframe = frame
            # # print(len(all_frames))
            # reduced_frames = []
            # req = 40
            
            # i = 0.0
            # if len(all_frames)>20:
            #     while i < len(all_frames) - len(all_frames)/req:
            #         reduced_frames.append(all_frames[int(i)])
            #         # cv2.imshow('frame',reduced_frames[-1])
            #         # if cv2.waitKey(1) & 0xFF == ord('q'): break 
            #         i += len(all_frames)/req
                
            #     while len(reduced_frames)< req:
            #         reduced_frames.append(reduced_frames[-1])
            #     # print("Red ",len(reduced_frames))
            #     ic(len(reduced_frames))
            #     labels.append(tag)
            #     processed_videos.append(reduced_frames)
            #     break

ic(np.array(processed_videos).shape)





dataset = []
data_loc = "/Volumes/Nantha/Workspace/Datasets/Nanthak Dataset/"
data_loc = "drive/MyDrive/My Dataset/"
labels = []
processed_videos = []

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

        for i in range(fcount//n,fcount,fcount//n):
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

            cap.set(1,i+fcount//n-1)
            framea[framea < 30] = 0  #denoising

            diff = cv2.absdiff(frame,pframea)
            diff[diff<30] = 0
            
            # count = np.count_nonzero(diff)
            all_frames.append(diff)
            pframea = np.array(framea)

        if len(all_frames) > 8:
            while len(all_frames) < 10:
                all_frames.append(all_frames[-1])
            labels.append(tag)
            processed_videos.append(all_frames)
        
        for frm in all_frames:
            cv2.show('frame',frm)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'): break 



        
get_data_nframes("drive/MyDrive/Dataset2")
get_data_nframes("drive/MyDrive/Ram Dataset")
get_data_nframes("drive/MyDrive/Dataset1")
get_data_nframes("drive/MyDrive/My Dataset")

ic(np.array(processed_videos).shape)