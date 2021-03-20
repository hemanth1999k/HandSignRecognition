# from attensat_xl import *
from attensat import *
import time
import sys
import tqdm
import progressbar
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
print("Importation Finished")

class iterdata(torch.utils.data.IterableDataset):
    
    def __init__(self,start,end):
        super(iterdata).__init__()
        assert end > start, "end < start"
        self.start = start 
        self.end = end

        self.videos = pickle.load(open("../saved/2FG-V.pkl","rb"))
        self.labels = pickle.load(open("../saved/2FG-L.pkl","rb"))
        print("Loading Pickle Finished")
        self._preproces()
        self.images = []
        self.imglabels = []
        ic(len(self.images))
        ic(len(self.imglabels))
    def shuffle(self,x,indices):
        r = []
        for i in range(len(indices)):
            r.append(x[indices[i]])
        return r
     
    def _preproces(self):
        self.images = []
        frames_count = 0
        self.imglabels = []
        self.lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 300:
                    self.images.append(f)
                    self.imglabels.append(self.lab_dic.index(l))
                frames_count+=1
        self.images = np.array(self.images)
        indices = list(range(len(self.images)))
        np.random.shuffle(indices) 
        # ic(indices[0])
        # ic(self.imglabels[0])
        shuffle_images = np.array(self.shuffle(self.images,indices))
        shuffle_labels = np.array(self.shuffle(self.imglabels,indices))

        # Need to remove for attensat model
        # shuffle_images =  shuffle_images[:,np.newaxis,:,:] 

        # self.X = shuffle_images[:8*len(shuffle_images)//10] 
        # self.Y = shuffle_labels[:8*len(shuffle_images)//10] 
        # self.VX = shuffle_images[8*len(shuffle_images)//10:] 
        # self.VY = shuffle_labels[8*len(shuffle_images)//10:] 
        # self.X = torch.tensor(self.X,dtype=torch.float32)
        # self.VX = torch.tensor(self.VX,dtype=torch.float32)
        # self.Y =  torch.tensor(self.Y,dtype=torch.long) 
        # self.VY =  torch.tensor(self.VY,dtype=torch.long) 
        print("Preprocess Finished")


    
    def info(self):
        lab_dic = {}
        img_dic = {}
        self.imglabels = []
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 300:
                    self.images.append(f)
                    self.imglabels.append(l)
                    if l in img_dic.keys():
                        lab_dic[l].append(self.lab_dic.index(l))
                        img_dic[l].append(f)
                    else:
                        lab_dic[l] = [self.lab_dic.index(l)] 
                        img_dic[l] = [f]
        return img_dic,lab_dic
    
    def make_videos(self,x,n):
        data = []
        for lab in x.keys():
            f = x[lab]
            sub_data = []
            for i in range(n):
                one_c = []
                for j in range(i,len(f),n):
                    one_c.append(f[j]) 
                sub_data.append(one_c)
            min_len = len(sub_data[0])

            for m in sub_data:
                if min_len  > len(m):
                    min_len = len(m)
            
            trimmed = []
            for m in sub_data:
                trimmed.append(m[:min_len])
            trimmed = np.array(trimmed)
            # print("AA",trimmed.shape)
            trimmed = np.transpose(trimmed,(1,0,2,3))
            data.append(trimmed)
            
        out = data[0]
        for u in range(1,len(data)):
            kk = data[u]
            out = np.concatenate((out,kk),axis=0)
        print(out.shape)
        
        return out 
        out = np.array(data[0])
    
    def make_dataset(self,x,y,n):
        l = self.imglabels
        data = []
        for lab in x.keys():
            f = x[lab]
            ic(f)
            sub_data = []
            for i in range(n):
                one_c = []
                for j in range(i,len(f),n):
                    one_c.append(f[j]) 
                sub_data.append(one_c)
            min_len = len(sub_data[0])

            for m in sub_data:
                if min_len  > len(m):
                    min_len = len(m)
            
            trimmed = []
            for m in sub_data:
                trimmed.append(m[:min_len])
            trimmed = np.array(trimmed)
            print(trimmed.shape)
            data.append(np.array(trimmed).T)
        out = np.array(data[0])

        for i in range(1,len(data)):
            u = data[i] 
            out = np.concatenate((out,u),axis=0)
        print("Removing unwanted")
        finout = np.array([out[0]])
        finout_video = np.array(y[0]) 
        for i in tqdm.tqdm(range(1,out.shape[0])):                
            row = out[i]
            if (row[0] == row).sum() != n:
                pass
            else:
                finout = np.concatenate((finout,[row]),axis=0)
                finout_video = np.concatenate((finout_video,y[i]),axis=0)
                pass

        for row in finout:
            if (row[0] == row).sum() != n:
                print(row)
                pass
        print(finout.shape)
        finout = finout[:100*(finout.shape[0]//100) ]
        finout = finout.reshape((-1,100,n))

        finout_video = finout_video[:100*(finout_video.shape[0]//100) ]
        finout_video = finout_video.reshape((-1,100,n,128,128))
        print(finout_video.shape,finout.shape)
        self.xl_labels = finout
        self.xl_images = finout_video
        self.set_batchsize(25,n)
        
       
    def set_batchsize(self,bs,mem_size):
        x = self.xl_images.reshape((-1,bs,mem_size,128,128))
        y = self.xl_labels.reshape((-1,bs,mem_size))
        x = np.transpose(x,(0,2,1,3,4)) 
        y = np.transpose(y,(0,2,1))
        x = x.reshape(-1,bs,128,128)
        y = y.reshape(-1,bs)
        
        self.X = x[:9*len(x)//10] 
        self.Y = y[:9*len(y)//10] 


        self.VX = x[9*len(x)//10:] 
        self.VY = y[9*len(x)//10:] 
        
        self.X = torch.tensor(self.X,dtype=torch.float32)
        self.VX = torch.tensor(self.VX,dtype=torch.float32)
        
        self.Y =  torch.tensor(self.Y,dtype=torch.long) 
        self.VY =  torch.tensor(self.VY,dtype=torch.long) 
        ic(self.X.shape,self.Y.shape,self.VX.shape,self.VY.shape)
 
class Epoch_Stat:
    
    avg_acc = 0.0
    avg_loss =0.0
    count = 0

    def __init__(self):
        pass

    def reset(self):
        self.avg_acc = 0.0
        self.avg_loss =0.0
        self.count = 0

    def get_results(self):
        return {"acc":self.avg_acc/self.count,"loss":self.avg_loss/self.count}

    def new_data(self,acc,loss):
        self.avg_acc  += acc
        self.avg_loss += loss
        self.count += 1

class Train_Stat:
    h_loss = [] 
    h_acc = []
    vh_loss = []
    vh_acc = []
    def __init__(self):
        pass                        

    def new_epoch(self,acc,loss,vacc=0.0,vloss=0.0):
        self.h_acc.append(acc)
        self.h_loss.append(loss)
        self.vh_acc.append(vacc)
        self.vh_loss.append(v)
    
    def new_epoch(self,train_dict,val_dict={"acc":None,"loss":None}):
        
        self.h_acc.append(   train_dict["acc"])
        self.h_loss.append(  train_dict["loss"])
        self.vh_acc.append(  val_dict[  "acc"])
        self.vh_loss.append( val_dict[  "loss"] )
 

def train_single_epoch(model,optim,loss_fn,it,mem_size):
    #batchsize is preset
    sat = Epoch_Stat()
    model.train()
    X,Y = it.X,it.Y

    pbar = tqdm.tqdm(range(X.shape[0]))
    for b in pbar: 
        
        x = X[b].detach().clone()
        y = Y[b].detach().clone()
        if b%mem_size == 0:
            # out = model(x,True)
            out = model(x)
        else:
            # out = model(x,True)
            out = model(x)

        loss = loss_fn(out,y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        acc = (torch.argmax(out,-1)==y).sum().float()/x.shape[0]
        pbar.set_description("A:"+str(acc.item())[:5]+" L:"+str(loss.item())[:5])
        sat.new_data(acc.item(),loss.item())
    return sat.get_results()

def train_val(model,loss_fn,it,mem_size):
    #batchsize is preset
    sat = Epoch_Stat()
    model.train()
    X,Y = it.VX,it.VY
    pbar = tqdm.tqdm(range(X.shape[0]))
    for b in pbar: 
        
        x = X[b].detach().clone()
        y = Y[b].detach().clone()
        
        if b%mem_size == 0:
            # out = model(x,True)
            out = model(x)
        else:
            # out = model(x,True)
            out = model(x)

        loss = loss_fn(out,y)
        acc = (torch.argmax(out,-1)==y).sum().float()/x.shape[0]
        pbar.set_description("A:"+str(acc.item())[:5]+" L:"+str(loss.item())[:5])
        sat.new_data(acc.item(),loss.item())
    return sat.get_results()

      

def train(model,epochs,learning_rate,memory_size=5):
    iterator = iterdata(0,1)
    imd,ld= iterator.info()
    vid = iterator.make_videos(imd,memory_size)
    iterator.make_dataset(ld,vid,memory_size)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
     
    loss_fn = torch.nn.CrossEntropyLoss()
    train_stats = Train_Stat()
    for e in range(epochs):
        train_dict = train_single_epoch(model,optim,loss_fn,iterator,memory_size)        
        
        if e%3 == 0:
            val_dict = train_val(model,loss_fn,iterator,memory_size)
            train_stats.new_epoch(train_dict,val_dict)
        else:
            train_stats.new_epoch(train_dict)
    return train_stats

if __name__ == '__main__':
    # iterator = iterdata(0,1)
    # imd,ld= iterator.info()
    # ms = 7
    # vid = iterator.make_videos(imd,ms)
    # iterator.make_dataset(ld,vid,ms)

    memory_size = 5
    iterator = iterdata(0,1)
    imd,ld= iterator.info()
    vid = iterator.make_videos(imd,memory_size)
    iterator.make_dataset(ld,vid,memory_size)
    ic(iterator.Y)
    for a in iterator.Y:
        ic(a)
 
    # model = AttenMod()
    # train(model,4,0.0001,memory_size=5)
    pass