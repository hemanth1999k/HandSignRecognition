#import tensorflow as tf
#import numpy as np
#import keras
#from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
#from keras.layers import TimeDistributed
#from keras.models import Sequential
#from keras.utils import to_categorical
#import kerasncp as kncp
#import numpy as np
#import matplotlib.pyplot as plt
#import h5py

#class Model:
#	
#	def __init__(self,outputs,load_name=None):
#		if load_name != None:
#			self.model = keras.models.load(load_name)		
#		else:
#			self.model = Sequential()
#			sample_shape = (30,128,128,1);
#			self.model.add(Conv3D(32,kernel_size=(3,3,3),activation ='relu', kernel_initializer='he_uniform',input_shape=sample_shape))
#			s:lf.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#			self.model.add(BatchNormalization(center=True, scale=True))
#			self.model.add(Dropout(0.5))
#			self.model.add(Conv3D(64, kernel_size=(3,3, 3), activation='relu', kernel_initializer='he_uniform'))
#			self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#			self.model.add(BatchNormalization(center=True, scale=True))
#			self.model.add(Dropout(0.5))
#			self.model.add(Flatten())
#			self.model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
#			self.model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
#			self.model.add(Dense(outputs, activation='softmax'))
#			self.model.compile(loss='sparse_categorical_crossentropy',
#						optimizer = keras.optimizers.Adam(lr=0.001),
#						metrics=['accuracy'])
#		self.model.summary() 
#	
#	def train(self,X_train,targets_train,batch,epochs,validation_split):
#		history = self.model.fit(X_train, targets_train,
#            batch_size=batch,
#            epochs=epochs,
#            verbose=1,
#            validation_split=validation_split)
#		print(history.history.keys())
#		return [history.history['accuracy'],history.history['loss'],
#				history.history['val_accuracy'],history.history['val_loss']]
#
#	def save(self,name):
#		self.model.save(name)
#
#class ModelNCP:
#	def __init__(self,outputs,load_name=None):
#		if load_name != None:
#			self.model = keras.models.load(load_name)		
#		else:
#			self.model = Sequential()
#			sample_shape = (30,128,128,1);
#					
#
#		self.model.summary() 
#	
#	def train(self,X_train,targets_train,batch,epochs,validation_split):
#		history = self.model.fit(X_train, targets_train,
#            batch_size=batch,
#            epochs=epochs,
#            verbose=1,
#            validation_split=validation_split)
#		print(history.history.keys())
#		return [history.history['accuracy'],history.history['loss'],
#				history.history['val_accuracy'],history.history['val_loss']]
#
#	def save(self,name):
#		self.model.save(name)	
#		
#if __name__ == '__main__':
#	m = Model()
#
from torch import nn
from icecream import ic
import torch
import cv2
import numpy as np
from icecream import ic


class Model(nn.Module):
    def __init__(self, nclasses):
        super(Model, self).__init__()
        self.m = nn.Conv3d(1, 20, 3, stride=1)
        self.p = nn.MaxPool3d((3, 3, 3), stride=2)

        self.m1 = nn.Conv3d(20, 30, 3, stride=3)
        self.p1 = nn.MaxPool3d((3, 3, 3), stride=1)
        self.f_ch = 64
        self.m2 = nn.Conv3d(30, self.f_ch, 1, stride=1)
        self.p2 = nn.MaxPool3d((4, 3, 3), stride=1)

        encod = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.encoder = nn.TransformerEncoder(encod, num_layers=2)
        # if video length changed, the value below needs to be
        # changed

        self.f1 = nn.Linear(self.f_ch * 256, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, nclasses)

    def forward(self, x):
        r = self.m(x)
        r = self.p(r)

        # ic(r.shape)
        r = self.m1(r)
        r = self.p1(r)
        # ic(r.shape)

        r = self.m2(r)
        r = self.p2(r)

        # ic(r.shape)
        # print(r.shape)
        assert torch.numel(r) % (x.shape[0] * self.f_ch) == 0

        r = r.reshape((-1, self.f_ch, torch.numel(r) // (x.shape[0] * self.f_ch)))
#        print(r.shape)
        r = self.encoder(r)
        # ic("Enc out",r.shape)
        r = r.reshape(x.shape[0], -1)

        r = nn.functional.relu(self.f1(r))
        r = nn.functional.relu(self.f2(r))
        r = nn.functional.softmax(self.f3(r), dim=-1)
        return r


def get_data(data_loca):
    seen = []
    prev_tag = None
    not_worked =  0
    for file in os.listdir(data_loca):
        if file[0] == ".":
            continue
        cap = cv2.VideoCapture(data_loca + "/" + file)
        if "Copy" in file:
            name = file.split(" ")[2].split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            # ic(file)
        else:
            name = file.split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()

        print(tag,file)

        w = 128
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        suc, frame = cap.read()
        if not suc:
            continue
        pframe = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
        pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
        all_frames = []

        for i in range(1, fcount):
            x, frame = cap.read()
            if not x:
                continue
            frame = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame, pframe)
            diff[diff < 30] = 0
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
        if len(all_frames) > 20:
            while i < len(all_frames) - len(all_frames) / req:
                reduced_frames.append(all_frames[int(i)])
                # cv2.imshow('frame',reduced_frames[-1])
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
                i += len(all_frames) / req

            while len(reduced_frames) < req:
                reduced_frames.append(reduced_frames[-1])
            # print("Red ",len(reduced_frames))
            ic(len(reduced_frames))
            labels.append(tag)
            seen.append(tag)
            processed_videos.append(reduced_frames)
        else:
            not_worked+=1
    return not_worked

def get_from_array(data_loca):
    # for folder in os.listdir(data_loc):
    #     if folder[0] != '.':
    seen = []
    prev_tag = None
    not_worked =  0
    for file in os.listdir(data_loca):
        if file[0] == ".":
            continue
        cap = cv2.VideoCapture(data_loca + "/" + file)
        if "Copy" in file:
            name = file.split(" ")[2].split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            # ic(file)
        else:
            name = file.split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()

        print(tag,file)



def get_from_array(video):
    w = 128
    frame = video[0]
    fcount = len(video) 
    pframe = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
    pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)

    all_frames = []

    for i in range(1, fcount):
        frame = video[i]
        frame = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, pframe)
        diff[diff < 30] = 0
        count = np.count_nonzero(diff)
        if count > 10:
            all_frames.append(diff)
        pframe = frame
    # print(len(all_frames))
    reduced_frames = []
    req = 40

    i = 0.0
    if len(all_frames) > 20:
        while i < len(all_frames) - len(all_frames) / req:
            reduced_frames.append(all_frames[int(i)])
            i += len(all_frames) / req

        while len(reduced_frames) < req:
            reduced_frames.append(reduced_frames[-1])
        ic(len(reduced_frames))
        return np.array(reduced_frames)
    else:			
        return np.array([])
