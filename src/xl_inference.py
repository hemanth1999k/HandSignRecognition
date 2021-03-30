import numpy as np
import time
import cv2 
import torch 
# import pygame as py
import sys
import matplotlib 
import random
from itertools import count
from attensat_xl import *
from atten_full_xl import *
import time
from interface import *
from torchsummary import summary
import matplotlib.pyplot as plt

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


class Model:
    def __init__(self,nback=5):
        self.model = AttenModXL()
        self.model.load_state_dict(torch.load("../models/divaten-X"+str(nback) + "L-SD.pth",map_location = torch.device('cpu')))
        self.model.eval()
        self.save_output = SaveOutput()
        self.handles = []
        self.index = 0
        self.nback = nback
        pass
    
    def change_model(self,mark,nback):
        if mark == 0:
            self.model = AttenModXL()
            model_name = "../models/divaten-X"+str(nback) + "L-SD.pth"
            ic("Loaded model ",model_name)
            self.model.load_state_dict(torch.load(model_name,map_location = torch.device('cpu')))
            
        if mark == 1:
            self.model = AttenModFullXL()
            model_name= "../models/FullX"+str(nback)+ "L.pth"
            ic("Loaded model ",model_name)
            self.model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu")))

        if mark == 2:
            self.model = AttenModFullXL()
            model_name= "../models/ContX"+str(nback)+ "L.pth"
            ic("Loaded model ",model_name)
            self.model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu")))
            
    def recognize(self,frame,capture_output=True):
        frame = frame[np.newaxis,:,:]
        # print("Recognize Called")
        frame = torch.tensor(frame,dtype=torch.float32)
        out = self.model(frame)    
        self.model.pop_memory(self.nback) 
        # self.plot_outputs()
        # self.save_output.clear()
        return out
        
    def plot_outputs(self):
        fig,axs = plt.subplots(8,32)
        w=32
        h=8
        fig=plt.figure(figsize=(8,8))
        columns =32 
        rows =8  
            
        for i in range(1, columns*rows +1):
            img = np.random.randint(10, size=(h,w))
            ax = fig.add_subplot(rows, columns, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            mat = np.array(self.save_output.outputs[1].reshape(-1,13,13)[i-1+4].detach().numpy()*255,dtype=np.uint8)
            plt.imshow(mat)
        plt.savefig('../plots/features'+str(self.index)+'.png',dpi=80)
        self.index+=1
                
class Recognize:
    def __init__(self,model):
        self.cap = cv2.VideoCapture(0)
        x = False
        self.model = model
        self.lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        self.last_f = None
        self.diffed = None
        self.confidence = None
        
    def capture_video(self,videopath):
        self.cap = cv2.VideoCapture(videopath)
        framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while framecount > 0:
            framecount-=1
            x,frame = self.cap.read()

            if x:
                s = 128
                frame = cv2.resize(frame,(s,s),0,0,cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if type(self.last_f) == type(None) and x:
                    self.last_f = np.array(frame,dtype="uint8")

                if type(self.last_f) != type(None):
                    self.diffed = cv2.absdiff(frame,self.last_f)
                    self.diffed[self.diffed<30] =0 
                    count = np.count_nonzero(self.diffed)
                    if count  > 10:
                        # cv2.imshow('frame',self.diffed)	
                        stime = time.time()
                        # cv2.imshow("frame",np.array(recog.diffed,dtype=np.uint8))
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break             
                        classify = torch.nn.functional.softmax(self.model.recognize(self.diffed),dim=-1)
                        etime = time.time()-stime
                        classify = classify.detach().numpy()
                        classify = classify[0].tolist() 
                        ans = [] 
                        for av in classify:
                            ans.append(format(av,".1f"))
                        sys.stdout.write('\r')
                        sys.stdout.write(self.lab_dic[np.argmax(classify)]+ " Time for calculation: "+str(etime))
                        sys.stdout.flush()
                    self.last_f = np.array(frame,dtype="uint8")
                    pass
      
    def capture(self):
        x,frame = self.cap.read()
        if x:
            s = 128
            frame = cv2.resize(frame,(s,s),0,0,cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            if type(self.last_f) == type(None) and x:
                self.last_f = np.array(frame,dtype="uint8")

            if type(self.last_f) != type(None):
                self.diffed = cv2.absdiff(frame,self.last_f)
                self.diffed[self.diffed<30] =0 
                count = np.count_nonzero(self.diffed)
                if count  > 10:
                    # cv2.imshow('frame',self.diffed)	
                    stime = time.time()
                    classify = torch.nn.functional.softmax(self.model.recognize(self.diffed),dim=-1)
                    self.confidence = (classify.detach().numpy(),self.lab_dic)
                    etime = time.time()-stime
                    classify = classify.detach().numpy()
                    classify = classify[0].tolist() 
                    ans = [] 
                    for av in classify:
                        ans.append(format(av,".1f"))

                    sys.stdout.write('\r')
                    sys.stdout.write(self.lab_dic[np.argmax(classify)]+ " Time for calculation: "+str(etime))
                    sys.stdout.flush()
                self.last_f = np.array(frame,dtype="uint8")
                pass
      
if __name__ == '__main__':
    model = Model(10)
    recog = Recognize(model)
    iF = Display()
    while True:
        recog.capture()
        commands = iF.update(recog.diffed,recog.confidence)
        if commands != None:
            model.change_model(*commands)
            recog.model = model

        # cv2.imshow("frame",recog.diffed)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break 