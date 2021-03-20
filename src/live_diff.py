import time
import math
k = time.time()
import torch 
import numpy as np
import cv2
import os
import pandas
#from model import *
from matplotlib import pyplot as plt


root_dir = "../dataset/"
dataset = []
all_tags = set()
count = 0;
processed_frames = []
for folder in os.listdir(root_dir):
	if folder[0] == '.':
		continue;

	for filename in os.listdir(root_dir+"/"+folder):
		if filename[0] != '.':
			if filename.split('_')[0] == 'v':
				tag = filename.split('_')[1]
			else: tag = filename.split('_')[0] 		

			print(folder,filename)



cap  = cv2.VideoCapture(0)
wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
x,fr = cap.read()
print(fr)


last_f = None
while True:
	x,frame = cap.read()
	if x:
		s = 128
		
		frame = cv2.resize(frame,(s,s),0,0,cv2.INTER_CUBIC)
		print(frame)
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



		if type(last_f) == type(None) and x:
			last_f = np.array(frame,dtype="uint8")

		if type(last_f) != type(None):
			diffed = cv2.absdiff(frame,last_f)
			diffed[diffed<30] =0 
			count = np.count_nonzero(diffed)
			print(count)
			if count  > 100:
				cv2.imshow('frame',diffed)	
			last_f = np.array(frame,dtype="uint8")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
exit()
#	



while True:
		
	#cap = cv2.VideoCapture(root_dir+"/"+folder+"/"+filename)    
	cap  = cv2.VideoCapture(0)
	wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("Ratio ",wid/high)
	ratio = wid/high
	width = 128 
	height = 128 	
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width);
	
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height);
	

	total_white_pixels_in_video_sequence = 0
	images = []
	framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.set(1,1)

	s,last_f = cap.read()
	#last_f = cv2.cvtColor(last_f,cv2.COLOR_BGR2GRAY)
	print(last_f)
	last_f = cv2.resize(last_f,(width,height),0,0,cv2.INTER_CUBIC)

	red_img = np.zeros((128,128,3),dtype="uint8")
	green =	np.zeros((128,128,3),dtype="uint8")
	blue = np.zeros((128,128,3),dtype="uint8")
	 
	red_img[:,:,2] = last_f[:,:,2]
	green[:,:,1] = last_f[:,:,1]
	blue[:,:,0] = last_f[:,:,0]
	
	frame = red_img
	frame = np.concatenate((red_img,green,blue),axis=1)
	last_f = frame

	for p in range(2,framecount):
		cap.set(1,p)
		s,frame = cap.read()	
		print(frame.shape)
		if type(frame) != type(None):
			#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			frame = cv2.resize(frame,(width,height),0,0,cv2.INTER_CUBIC)

			red_img = np.zeros((128,128,3),dtype="uint8")
			green = np.zeros((128,128,3),dtype="uint8")
			blue = np.zeros((128,128,3),dtype="uint8")
			 
			red_img[:,:,2] = frame[:,:,2]
			green[:,:,1] = frame[:,:,1]
			blue[:,:,0] = frame[:,:,0]
		
			frame = red_img
			frame = np.concatenate((red_img,green,blue),axis=1)

			#processed_frames.append(np.abs(frame + frame-last_f))
			
			diffed = cv2.absdiff(frame,last_f)
			diffed[diffed<50] = 0
			#if np.count_nonzero(diffed) > 5000:
			processed_frames.append(diffed)
			#processed_frames.append(frame)
			if len(processed_frames) > 0:
				cv2.imshow('frame',processed_frames[-1])
			#print(processed_frames[-1])
			last_f = np.array(frame)	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			print(len(processed_frames))
			for p in processed_frames:
				cv2.imshow('frame',p)
			break;

#while True:
#	for p in processed_frames:
#		cv2.imshow('frame',p)
#		time.sleep(0.1)
#		if cv2.waitKey(1) & 0xFF == ord('q'):
#
dataset = []
data_loc = "../dataset/"
for folder in os.listdir(data_loc):
    if folder[0] != '.':
        for file in os.listdir(data_loc+folder):
            cap = cv2.VideoCapture(data_loc + folder + "/"+ file)
            w = 128
            fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            suc,frame = cap.read()
            pframe = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
            pframe = cv2.cvtColor(pframe,cv2.COLOR_BGR2GRAY)
            all_frames = []

            for i in range(1,fcount):
                x,frame = cap.read()
                frame = cv2.resize(frame,(w,w),0,0,cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(frame,pframe)
                diff[diff<30] = 0
                count = np.count_nonzero(diff)
                if count > 10:
                    # cv2.imshow('frame',diff)
                    # if cv2.waitKey(1) & 0xFF == ord('q'): break 
                    all_frames.append(diff)
                pframe = frame
            # print(len(all_frames))
            reduced_frames = []
            req = 40
            
            i = 0.0
            while i < len(all_frames) - len(all_frames)/req:
                reduced_frames.append(all_frames[int(i)])
                cv2.imshow('frame',reduced_frames[-1])
                if cv2.waitKey(1) & 0xFF == ord('q'): break 
                i += len(all_frames)/req
            print(len(reduced_frames))
            while len(reduced_frames)< req:
                reduced_frames.append(reduced_frames[-1])
            print("Red ",len(reduced_frames))
                
            break
            