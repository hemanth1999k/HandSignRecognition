import torch
from atten_full_xl import * 
import time
import sys
import tqdm
import pickle
from icecream import ic
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

        
        # self.videos = pickle.load(open("drive/MyDrive/2FG-V.pkl","rb"))
        # self.labels = pickle.load(open("drive/MyDrive/2FG-L.pkl","rb"))


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

    def xl_preprocessing(self,bs):
        self.images = []
        frames_count = 0
        self.imglabels = []
        self.lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 100:
                    self.images.append(f)
                    self.imglabels.append(self.lab_dic.index(l))
                frames_count+=1
        print()
        ic(len(self.images),len(self.imglabels))
        self.images = np.array(self.images)[:-(len(self.images)%bs),:,:]
        self.imglabels = np.array(self.imglabels)[:-(len(self.imglabels)%bs)] 
        
        self.images = self.images.reshape((bs,-1,128,128))
        self.imglabels = self.imglabels.reshape((bs,-1))
        ic(self.images.shape)
        ic(self.imglabels.shape)
        
        self.X = torch.tensor(self.images[:,:int(0.8*self.images.shape[1]) ] )
        self.Y = torch.tensor(self.imglabels[:,:int(0.8*self.imglabels.shape[1] )])

        self.VX = torch.tensor(self.images[:,int(0.8*(self.images.shape[1])):] )
        self.VY = torch.tensor(self.imglabels[:,int(0.8*(self.imglabels.shape[1])):] )

        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.VX = self.VX.cuda()
            self.VY = self.VY.cuda()

        ic(self.X.shape)
        ic(self.VX.shape)
        ic(self.Y.shape)
        ic(self.VY.shape)

    def _preproces(self):
        self.images = []
        frames_count = 0
        self.imglabels = []
        self.lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 100:
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
                for j in range(i,(len(f)//n)*n,n):
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
            # ic(f)
            sub_data = []
            for i in range(n):
                one_c = []
                for j in range(i,(len(f)//n)*n,n):
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
            # print(trimmed.shape)
            data.append(np.array(trimmed).T)
        out = np.array(data[0])

        for i in range(1,len(data)):
            u = data[i] 
            out = np.concatenate((out,u),axis=0)
            
        print("Removing unwanted")
        finout = np.array(out)
        # finout_video = np.array(y[0]) 
        finout_video = np.array(y)
        for row in finout:
            if (row[0] == row).sum() != n:
                # print(row)
                pass
        
        print(finout.shape)
        finout = finout[:100*(finout.shape[0]//100) ]
        finout = finout.reshape((-1,100,n))

        finout_video = finout_video[:100*(finout_video.shape[0]//100) ]
        finout_video = finout_video.reshape((-1,100,n,128,128))
        print(finout_video.shape,finout.shape)

        self.xl_labels = torch.tensor(finout)
        self.xl_images = torch.tensor(finout_video)
        self.set_batchsize(100,n)
        
       
    def set_batchsize(self,bs,mem_size):
        self.xl_images = self.xl_images.reshape((-1,mem_size,128,128))
        self.xl_labels = self.xl_labels.reshape((-1,mem_size))

        perm = torch.randperm(self.xl_labels.shape[0])

        self.xl_images = self.xl_images[perm]
        self.xl_labels = self.xl_labels[perm]

        x = self.xl_images.reshape((-1,bs,mem_size,128,128))
        y = self.xl_labels.reshape((-1,bs,mem_size))

        x = torch.transpose(x,1,2)
        y = torch.transpose(y,1,2)

        # x = np.transpose(x,(0,2,1,3,4)) 
        # y = np.transpose(y,(0,2,1))

        x = x.reshape(-1,bs,128,128)
        y = y.reshape(-1,bs)
        
        self.X = x[:8*len(x)//10]
        self.Y = y[:8*len(y)//10] 
        self.VX = x[8*len(x)//10:] 
        self.VY = y[8*len(x)//10:] 
        if torch.cuda.is_available(): 
            self.X = torch.tensor(self.X,dtype=torch.float32).cuda()
            self.VX = torch.tensor(self.VX,dtype=torch.float32).cuda()
        
            self.Y =  torch.tensor(self.Y,dtype=torch.long).cuda()
            self.VY =  torch.tensor(self.VY,dtype=torch.long).cuda()
        else:
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
        acc = self.avg_acc/self.count 
        loss= self.avg_loss/self.count
        ic(acc,loss)
        return {"acc":acc,"loss":loss}

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
        ic(train_dict)
        if val_dict["acc"] != None:
            ic(val_dict)
        self.h_acc.append(   train_dict["acc"])
        self.h_loss.append(  train_dict["loss"])
        self.vh_acc.append(  val_dict[  "acc"])
        self.vh_loss.append( val_dict[  "loss"] )
 
def train_single_batch(model,optim,loss_fn,it,mem_size):
    sat = Epoch_Stat()
    model.train()
    X,Y = it.X,it.Y

    pbar = tqdm.tqdm(range(0,mem_size),position=0,leave=True)
    for b in pbar: 
        x = X[b].clone().detach()
        y = Y[b].clone().detach()

        if b%mem_size == 0:
            out = model(x,True)
            # out = model(x)
        else:
            out = model(x)
            # out = model(x,True)

        loss = loss_fn(out,y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        acc = (torch.argmax(out,-1)==y).sum().float()/x.shape[0]
        pbar.set_description("A:"+str(acc.item())[:5]+" L:"+str(loss.item())[:5])
        sat.new_data(acc.item(),loss.item())
    
    return sat.get_results()




def train_single_epoch(model,optim,loss_fn,it,mem_size):
    #batchsize is preset
    sat = Epoch_Stat()
    model.train()
    X,Y = it.X,it.Y

    pbar = tqdm.tqdm(range(0,X.shape[1]),position=0,leave=True)
    for b in pbar: 
        x = X[:,b].clone().detach()
        y = Y[:,b].clone().detach()

        if b%mem_size == 0:
            model.pop_memory(mem_size)
            out = model(x)
            # out = model(x)
        else:
            out = model(x)
            # out = model(x,True)
        
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
    model.eval()
    X,Y = it.VX,it.VY
    # pbar = tqdm.tqdm(range(X.shape[0]))
    with torch.no_grad():
        for b in tqdm.tqdm(range(X.shape[0]),leave=True,position=0):
            x = X[b].detach().clone()
            y = Y[b].detach().clone()
            out = model(x)
            # print(y)

            if b%mem_size == 0:
                out = model(x,True)
                # out = model(x)
            else:
                # out = model(x,True)
                out = model(x)

            loss = loss_fn(out,y)
            acc = (torch.argmax(out,-1)==y).sum().float()/x.shape[0]
            # pbar.set_description("A:"+str(acc.item())[:5]+" L:"+str(loss.item())[:5])
            sat.new_data(acc.item(),loss.item())
    return sat.get_results()

def trainXL(model,epochs,learning_rate,memory_size=5):
    iterator = iterdata(0,1)
    iterator.xl_preprocessing(5)
    optim = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
     
    train_stats = Train_Stat()
    for e in range(epochs):
        train_dict = train_single_epoch(model,optim,loss_fn,iterator,memory_size)
    
    
def train(model,epochs,learning_rate,memory_size=5):
    iterator = iterdata(0,1)
    imd,ld= iterator.info()
    vid = iterator.make_videos(imd,memory_size)
    iterator.make_dataset(ld,vid,memory_size)
    optim = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
     
    train_stats = Train_Stat()
    for e in range(epochs):
        train_dict = train_single_batch(model,optim,loss_fn,iterator,memory_size)
        # if e%3 == 0:
        #     val_dict = train_val(model,loss_fn,iterator,memory_size)
        #     train_stats.new_epoch(train_dict,val_dict)
        # else:
        #     train_stats.new_epoch(train_dict)
    return train_stats



# if __name__ == '__main__':
#     model = AttenModFullXL()
#     memory_size=5
#     mem_size=5
#     train(model,40,0.01,memory_size=5)
#     pass

if __name__ == '__main__':
    model = AttenModFullXL()
    msize = 3 
    trainXL(model,2,0.01,memory_size=msize)