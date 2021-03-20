import time
import math
k = time.time()

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


data = pandas.DataFrame();

labels = []

for folder in os.listdir(root_dir):
	if folder[0] == '.':
		continue
	for filename in os.listdir(root_dir+"/"+folder):
		if filename[0] != '.':
			if filename.split('_')[0] == 'v':
				tag = filename.split('_')[1]
			else: tag = filename.split('_')[0] 		
			all_tags.add(tag)
			labels.append(tag)
			cap = cv2.VideoCapture(root_dir+"/"+folder+"/"+filename)    
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));

			cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128);
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT,128);
			total_white_pixels_in_video_sequence = 0
			images = []
			framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			j = 0
			required_count = 30
			while j < framecount:
				i = math.ceil(j)				
				j += framecount/required_count
				cap.set(1,i)
				success, frame = cap.read()
				if(type(frame) != type(None)):
					frame = cv2.resize(frame, (128,128), 0, 0, cv2.INTER_CUBIC);
										#redchannel = frame[:,:,2]
					#frame = frame[:,:,1]
					
					print(w,h)

					red_img = np.zeros((128,128,3),dtype="uint8")
					green = np.zeros((128,128,3),dtype="uint8")
					blue = np.zeros((128,128,3),dtype="uint8")
					 
					red_img[:,:,2] = frame[:,:,2]
					green[:,:,1] = frame[:,:,1]
					blue[:,:,0] = frame[:,:,0]
				
					frame = red_img
					frame = np.concatenate((red_img,green,blue),axis=1)
					print(frame.shape)
					#red_img = np.zeros(frame.shape)
					#red_img[:,:,2] = redchannel
					#print(frame.shape)
					#gray = frame[0].reshape(128,128)
					
					#gray = cv2.cvtColor(frame[0],cv2.COLOR_BGR2GRAY)
					#gray = red_img;	
					print(frame)
					gray = frame
					if(len(images)<required_count):
						images.append(gray)	
					count+=1
					cv2.imshow('frame',gray);
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
			images = np.array(images)
			dataset.append(images)
			cap.release()
			cv2.destroyAllWindows()

print(len(dataset),len(labels))

all_tags = list(all_tags)
Y = []
for x in labels:
	Y.append(all_tags.index(x)) 
Y = np.array(Y)

data["labels"] = Y
data["frames"] = dataset

print(data["labels"])
print(data["frames"][0].shape)

m = Model(len(all_tags))
#dataset = np.array(dataset).reshape(-1,30,128,128,1)
#
#print(dataset.shape,Y.shape)
#train = False
#save = False
#if train:
#	items = m.train(dataset,Y,6,10,0.3)
#	
#	
#	plt.plot(items[0])
#	plt.plot(items[2])
#	plt.title('model accuracy')
#	plt.ylabel('accuracy')
#	plt.xlabel('epoch')
#	plt.legend(['train', 'test'], loc='upper left')
#	plt.show()
#	
#	plt.plot(items[1])
#	plt.plot(items[3])
#	plt.title('model loss')
#	plt.ylabel('loss')
#	plt.xlabel('epoch')
#	plt.legend(['train', 'test'], loc='upper left')
#	plt.show()
#if save:	
#	m.save("../models/baby")
#
#
#
#
