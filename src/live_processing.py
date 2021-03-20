import numpy as np
import time
import cv2 
import torch 
# import pygame as py
import sys
import matplotlib 
import random
from itertools import count
from attensat import *
import time
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
    def __init__(self):
        self.model = torch.load("../models/divaten_model_atten_apen.pth",map_location = torch.device('cpu'))
        self.save_output = SaveOutput()
        self.handles = []
        self.index = 0
        for layer in self.model.modules():
            if isinstance(layer,torch.nn.modules.conv.Conv2d):
                hooks = layer.register_forward_hook(self.save_output)
                self.handles.append(hooks)
                
        pass
    
    def recognize(self,frame,capture_output=True):
        frame = frame[np.newaxis,:,:]
        print("Recognize Called")
        frame = torch.tensor(frame,dtype=torch.float32)
        out = self.model(frame)    

        self.print_output()
        self.plot_outputs()
        self.save_output.clear()
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
                
    def print_output(self):
        print("Printing Output")
        for i in self.save_output.outputs:
            ic(i.shape)


        pass
# class Model:
#     def __init__(self):
#         self.model = torch.load("../models/divaten_model_atten_apen.pth",map_location = torch.device('cpu'))

#     def recognize(self,frame):
#         frame = frame[np.newaxis,:,:]
#         frame = torch.tensor(frame,dtype=torch.float32)
#         out = self.model(frame)    
#         return out


data_stored =  []
for i in range(25):
    data_stored.append([]) 

class Recognize:
    def __init__(self,model):
        # self.cap = cv2.VideoCapture(0)
        self.cap = None
        x = False
        # while not x:
        #     x,fr = self.cap.read()
        # print(fr)
        self.model = model
        self.lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        self.last_f = None
        self.diffed = None
        
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
                        cv2.imshow("frame",np.array(recog.diffed,dtype=np.uint8))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break             
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
                    cv2.imshow('frame',self.diffed)	
                    stime = time.time()
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
       
def shot_video(model):
    
    cap  = cv2.VideoCapture(0)
    wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x,fr = cap.read()
    # print(fr)
    lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
    last_f = None
    while True:
        x,frame = cap.read()
        if x:
            s = 128
    
            frame = cv2.resize(frame,(s,s),0,0,cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            if type(last_f) == type(None) and x:
                last_f = np.array(frame,dtype="uint8")

            if type(last_f) != type(None):
                diffed = cv2.absdiff(frame,last_f)
                diffed[diffed<30] =0 
                count = np.count_nonzero(diffed)
                if count  > 10:
                    cv2.imshow('frame',diffed)	

                    classify = torch.nn.functional.softmax(model.recognize(diffed),dim=-1)
                    classify = classify.detach().numpy()
                    for i in range(len(classify[0])):
                        data_stored[i].append(classify[0][i])
                    if len(data_stored[0]) > 20:
                        for i in range(25):
                            data_stored[i].pop(0)
                        # plt.cla()
                        
                    for i in data_stored:
                        # plt.plot(list(range(len(i))),i,linestyle='--') 
                        pass

                    classify = classify[0].tolist() 
                    ans = [] 
                    for av in classify:
                        ans.append(format(av,".1f"))

                    sys.stdout.write('\r')
                    sys.stdout.write(lab_dic[np.argmax(classify)])
                    sys.stdout.flush()


                    
                    # fig = plt.figure()
                    # ax = fig.add_axes([0,0,1,1])
                    # ax.bar(lab_dic,classify)
                    # ax.set_ylim(0,1)
                    # plt.show()gg
                    # plt.legend(lab_dic,loc="upper left")
                    # plt.show()
                    # plt.pause(0.0001)
                   # plt.title('Dynamic line graphs')
                    # ind = np.argmax(classify);v = classify[ind];classify[ind] = 0
                    # ind1 = np.argmax(classify);v1 = classify[ind1];classify[ind1] = 0
                    # ind2 = np.argmax(classify);v2 = classify[ind2];classify[ind2] = 0
                    # print("-------------------------------------")
                    # print(lab_dic[ind], " : ",v)
                    # print(lab_dic[ind1]," : ",v1)
                    # print(lab_dic[ind2]," : ",v2)

                last_f = np.array(frame,dtype="uint8")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                                               

def plot_attention(name,attention):
    # print("Att shape",attention.shape)
    # ax = plt.gca()

    if len(attention) == 16:
        fig = plt.figure(dpi=30)
        for i in range(1,17):
            ax = fig.add_subplot(4,4,i)
            ax.matshow(attention[i-1][0])
    #     # plt.show()
        plt.pause(0.5)
        plt.close()
    print("Plotted")
        
import threading
if __name__ == '__main__':
    model = Model() 
    # t1 = threading.Thread(target=plotting)    
    recog = Recognize(model)
    # threads = []
    recog.capture_video("../dataset/2 book/v_book_c1.mp4")

    # c = 0

    # while 1:
    #     recog.capture()

    #     # plot_attention("A",recog.model.model.att_mat)
    #     cv2.imshow("frame",recog.diffed)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break             
    #     c+=1



    # for t in threads:
    #     t.join()
    # t1.join()
    # out= model.model(torch.ones((1,128,128)))
    # summary(model.model,input_size=(128,129)) 