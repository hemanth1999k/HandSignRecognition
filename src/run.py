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

        ic(len(self.images))
        ic(len(self.imglabels))

    def _preproces(self):
        self.images = []
        frames_count = 0
        self.imglabels = []
        lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 300:
                    self.images.append(f)
                    self.imglabels.append(lab_dic.index(l))
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

        self.X = shuffle_images[:8*len(shuffle_images)//10] 
        self.Y = shuffle_labels[:8*len(shuffle_images)//10] 
        self.VX = shuffle_images[8*len(shuffle_images)//10:] 
        self.VY = shuffle_labels[8*len(shuffle_images)//10:] 
        self.X = torch.tensor(self.X,dtype=torch.float32)
        self.VX = torch.tensor(self.VX,dtype=torch.float32)
        self.Y =  torch.tensor(self.Y,dtype=torch.long) 
        self.VY =  torch.tensor(self.VY,dtype=torch.long) 
        print("Preprocess Finished")

        
        # ic(self.X.shape,self.VX.shape) 

    def shuffle(self,x,indices):
        r = []
        for i in range(len(indices)):
            r.append(x[indices[i]])
        return r
    def __iter__(self):
        return iter(self.images[self.start:self.end])

    def __len__(self):
        return len(self.images)


def getBack(var_grad_fn,):
    # print(var_grad_fn)
    count = 0
    for n in var_grad_fn.next_functions:
        count+=1
        if n[0]:
            try:
                # tensor = getattr(n[0], 'variable')
                # print(n[0].shape)
                # print('Tensor with grad found:', tensor.shape)
                # print(' - gradient:', tensor.grad.shape)
                count+=1
                print(n[0].shape)
            except AttributeError as e:
                count+=getBack(n[0])
    # ic("Grad count",count)
    return count


def train(model,epochs):
    it = iterdata(0,24971)

    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    h_loss = []
    h_acc = []
    vh_loss = []
    vh_acc = []
    
    for e in range(epochs):
        avg_acc = 0.0
        avg_loss = 0.0
        c = 0
        model.train()
        bs = 10
        for i in tqdm.tqdm(range(0,100,bs)):
            model.train()
            X = it.X[i:i+bs].clone().detach()
            Y = it.Y[i:i+bs].clone().detach()
            print(X.shape,Y.shape)
            inp_time = time.time()
            y = model(X)
            print("Input time",time.time() - inp_time)
            # ic(y.shape,Y.shape)
            loss = loss_fn(y,Y)
            avg_loss += loss.item()
            back_time = time.time()
            loss.backward()
            print(X.grad)
            # ic("Entering print graph")

            # grad_count = getBack(loss.grad_fn)
            # ic("Exiting print graph",grad_count)
            ic(time.time()-back_time)
            step_time = time.time()
            optim.step()

            optim.zero_grad()
            # print(dir(loss))
            # loss.zero_()
            ic(time.time()-step_time)
            accuracy = (torch.argmax(y,-1)==Y).sum().float()/X.shape[0]
            print(accuracy.grad)
            avg_acc += accuracy
            c+=1
            # break
            # sys.stdout.write('\r')
            # # the exact output you're looking for:
            # sys.stdout.write("[%-10s] %d%%" % ('='*int(i*10/len(it.X))), (i*100/len(it.X)))
            # sys.stdout.flush()

        avg_loss = avg_loss/c 
        avg_acc = avg_acc/c
        ic("Epoch ",e,avg_acc,avg_loss)
        h_acc.append(avg_acc) 
        h_loss.append(avg_loss)

        if e%4 == 0:
            avg_acc = 0.0
            avg_loss = 0.0
            c = 0
            model.eval()
            bs = 10 
            for i in range(0,100,bs):
                with torch.no_grad():
                    VX = it.VX[i:i+bs]
                    VY = it.VY[i:i+bs]
                    y = model(VX)
                    # ic(y.shape,VY.shape)
                    loss = loss_fn(y,VY)
                    avg_loss += loss.item()
                    accuracy = (torch.argmax(y,-1)==VY).sum().float()/VX.shape[0]
                    avg_acc += accuracy
                    c+=1
            
            vavg_loss = avg_loss/c 
            vavg_acc = avg_acc/c
            
            print("Epoch :",e,"Validation Accuracy :",vavg_acc,"Validation Loss :",vavg_loss)
            ic(vavg_acc,vavg_loss)
            vh_acc.append(vavg_acc)
            vh_loss.append(vavg_loss)
            ic(vavg_acc,max(vh_acc))
            if vavg_acc >= max(vh_acc):
                ic("Saving model",vavg_acc)
                torch.save(model,"../models/divaten_model.pth") 
                
            plt.clf()
            plt.plot(range(len(h_loss)), h_loss)
            plt.plot(range(0, len(vh_loss) * 4, 4), vh_loss)
            plt.xlabel("epochs")
            plt.ylabel("loss")
            # plt.show()
            plt.savefig("../plots/loss_1"+".png")
            plt.clf()
            plt.plot(range(len(h_acc)), h_acc)
            plt.plot(range(0, len(vh_acc) * 4, 4), vh_acc)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            #plt.show()
            plt.savefig("../plots/acc_1" + ".png")
            plt.clf()
            
    return model

                
if __name__ == '__main__':
    # # ds = torch.utils.data.DataLoader(it,num_workers=0)
    # # attensat = ConvModel()

    # u = torch.tensor(list(range(128*128*10)),dtype=torch.float32)
    # u = u.reshape((10,128,128))
    
    attensat = AttenMod()
    convmodel = ConvModel()
    
    # pytorch_total_params = sum(p.numel() for p in attensat.parameters()) 
    # pytorch_total_params1 = sum(p.numel() for p in convmodel.parameters()) 
    
    # # print(attensat)
    # ic(pytorch_total_params)
    # ic(pytorch_total_params1)
    # attensat(u) 
    
    trained_model = train(attensat,30)
    # trained_model = train(convmodel,30)

    u = torch.tensor(list(range(128*128*1)),dtype=torch.float32)
    u = u.reshape((1,128,128))

    # attensat = Attensat()
    # conv = ConvModel()

    import time
    x = time.time()
    result = attensat(u)
    print(time.time()-x)

    
    x = time.time()
    u = u.reshape((-1,1,128,128))
    result = convmodel(u)
    print(time.time()-x)
    print(result.shape)
    
    # result1 = conv(u.reshape((1,1,128,128)))
    # print(result1.shape)
