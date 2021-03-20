import torch
from torch import nn 
from icecream import ic
import csv
import os
import cv2
import numpy as np
dataset = []
data_loc = "../dataset/"
labels = []
processed_videos = []
for folder in os.listdir(data_loc):
    if folder[0] != '.':
        for file in os.listdir(data_loc+folder):
            cap = cv2.VideoCapture(data_loc + folder + "/"+ file)

            name = file.split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            

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
            labels.append(tag)
            processed_videos.append(reduced_frames)
            break

print(np.array(processed_videos).shape)
def transformY(y):
    av = list(set(labels))
    Y = []
    for x in y:
        Y.append(av.index(x))
    print(y)
    
    return np.array(Y)

Y = transformY(labels)
X = np.array(processed_videos)
X = X[:,np.newaxis,:,:,:]
print(X.shape,Y.shape)

X = torch.tensor(X,dtype=torch.float32)
Y = torch.tensor(Y,dtype=torch.fjnloat32)
X.shape

class Model(nn.Module):
    def __init__(self,nclasses):
        super(Model,self).__init__()
        self.m = nn.Conv3d(1,20,3,stride=1)
        self.p = nn.MaxPool3d((3,3,3),stride=2)
        self.m1 = nn.Conv3d(20,20,3,stride=1)
        self.p1 = nn.MaxPool3d((3,3,3),stride=3)
        self.m2 = nn.Conv3d(20,30,3,stride=1)
        self.p2 = nn.MaxPool3d((3,3,3),stride=2)
        
        encod = nn.TransformerEncoderLayer(d_model = 64,nhead=4)
        self.encoder = nn.TransformerEncoder(encod,num_layers=4)
        #if video length changed, the value below needs to be
        #changed
        vid_len = 30

        self.f1= nn.Linear(64*vid_len,48)
        self.f2 = nn.Linear(48,32)
        self.f3 = nn.Linear(32,nclasses)


    def forward(self,x):
        r = self.m(x)
        r = self.p(r)
        r = self.m1(r)
        r = self.p1(r)
        r = self.m2(r)
        r = self.p2(r)
        r = r.reshape((-1,30,64))
        r = self.encoder(r)

        r = r.reshape(x.shape[0],-1)
        
        r = nn.functional.relu(self.f1(r))
        r = nn.functional.relu(self.f2(r))
        r = nn.functional.softmax(self.f3(r),dim=-1)
        return r

model = Model(4)

optim = torch.optim.SGD(model.parameters(),lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
h_loss = []
h_acc = []
for epoch in range(20):
    y = model(X)
    loss = loss_fn(y,Y)
    loss.backward()
    
    accuracy = (torch.argmax(y,-1)==Y).sum().float()/X.shape[0]
    ic(loss.item(),accuracy.item())
    h_loss.append(loss.item())
    h_acc.append(accuracy.item())
    
    optim.step()
    optim.zero_grad()
 