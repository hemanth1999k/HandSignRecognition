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


print('caputuring video')
# cap  = cv2.VideoCapture(0)
# print(cap)
# wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(wid,high)
# x,fr = cap.read()
# print(fr)



# while True:
# 	x,fr = cap.read()
# 	if x:
# 		cv2.imshow('frame',fr)
# 	time.sleep(0.1)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
# #	



while True:
		
	cap = cv2.VideoCapture(root_dir+"/"+folder+"/"+filename)    
	# cap  = cv2.VideoCapture(0)
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
	last_f = frameee
	Knanthak kumar edit
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
#			break
