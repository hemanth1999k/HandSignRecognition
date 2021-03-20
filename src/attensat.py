import torch
import torch.nn as nn
from icecream import ic
import os

class Flatten(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.reshape((bs, -1))


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(57600, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 25),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
       #ic(x.shape)
        u = self.seq(x)
       #ic(u.shape)
        return u


class SeparateLinear(torch.nn.Module):
    def __init__(self, n, ins, out):
        super().__init__()

        self.Linears = {}
        for i in range(n):
            self.Linears[str(i)] = nn.Linear(ins, out)
        self.Linears = nn.ModuleDict(self.Linears)

    def forward(self, x):
        outs = []
        bs = x.shape[0]
        x = x.reshape((16,bs,32,6,6)) 
        for i in range(x.shape[0]):
            a = self.Linears[str(i)](x[i])
            outs.append(a)
        ou = torch.stack(outs)
        ou = ou.reshape(bs,16,32,6,16)
        ##ic(ou.shape)
        return ou


#divaten_model_atten_apen
class AttenMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(0.1)
        self.expand = nn.Linear(36,64)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.features = None
        dicts = {}
        for i in range(16):
            dicts["M-atten-" + str(i)] = nn.MultiheadAttention(64, 4, dropout=0.2)     
        
        self.multiheads = nn.ModuleDict(dicts)
        self.atten_dropout = nn.Dropout(0.2)
        
        self.layernorm1 = nn.LayerNorm(64)
        encod = nn.TransformerEncoderLayer(d_model=2048, nhead=16)
        self.final_encoder = nn.TransformerEncoder(encod, num_layers=1)
        
        self.flin1 = nn.Linear(16*2048,512)
        self.flin2 = nn.Linear(512,128)
        self.flin3 = nn.Linear(128,25)
        self.att_mat = None
        
 
    def forward(self,t):
        t = t/255.0 # normaliztion added to the model forward itself
        bs = t.shape[0]
        sliced = torch.zeros((t.shape[0],16, 32, 32))
        # ic(x.shape)
        for i in range(4):
            for j in range(4):
                # print(" {}:{}, {}:{}".format(i * 32, i * 32 + 32, j * 32, j * 32 + 32))
                # print(t.shape)
                sliced[:,i * 4 + j] = t[:,i * 32:i * 32 + 32, j * 32:j * 32 + 32]

        x = sliced
        x = x.reshape((bs*16, 1, 32, 32))
        
        u = self.conv1(x)
        u = self.pool1(u)
        self.features = u.detach().clone()
        u = self.dropout1(u)

        u = self.conv2(u)
        u = self.pool2(u)
        u = self.dropout2(u)
        u = u.reshape((bs*16,32,36))
        u = self.expand(u)         
        u = self.relu1(u)
        # ic(u.shape)
        u = u.reshape((16,32,bs,64))
        attention_out =[]
        self.att_mat = []
        for i in range(len(self.multiheads)):
            att_out, _att_mat = (self.multiheads["M-atten-" + str(i)](u[i] ,u[i], u[i]))
            self.att_mat.append(_att_mat.detach().clone())
            attention_out.append(att_out)
            
        with torch.no_grad():
            self.att_mat = torch.stack(self.att_mat)

        attention_out_stacked = torch.stack(attention_out) 
        u = u + self.atten_dropout(attention_out_stacked)
        u = self.layernorm1(u)
        u = u.reshape((16,bs,32*64))
        u = self.final_encoder(u)         
        u = u.reshape((bs,16*2048))
        u = self.flin1(u)
        u = nn.functional.relu(u)
        u = self.flin2(u)
        u = nn.functional.relu(u)
        u = self.flin3(u)
        # ic(u.shape)
        return u


class Attensat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.feat_encode = nn.Linear(6, 16)
        self.expand = SeparateLinear(16, 6, 16)
        encod = nn.TransformerEncoderLayer(d_model=2048, nhead=16)
        self.encoder = nn.TransformerEncoder(encod, num_layers=2)
        dicts = {}
        self.lin_bef_encod = nn.Linear(3072,2048)
        for i in range(16):
            dicts["M-atten-" + str(i)] = nn.MultiheadAttention(96, 1, dropout=0.2)

        
        self.minimultiheads = nn.ModuleDict(dicts)
        self.linear_norm1 = nn.LinearNorm(())
        self.full_con = nn.Sequential(
            nn.Linear(16*2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,25),
            nn.Softmax(dim = -1),
        )

    def forward(self, x):
        bs = x.shape[0]
        with torch.no_grad():
            sliced = torch.zeros((x.shape[0],16, 32, 32))
           #ic(x.shape)
            for i in range(4):
                for j in range(4):
                    # print(" {}:{}, {}:{}".format(i * 32, i * 32 + 32, j * 32, j * 32 + 32))
                    slice = x[:,i * 32:i * 32 + 32, j * 32:j * 32 + 32]
                    sliced[:,i * 4 + j] = slice

            x = sliced
            x = x.reshape((bs*16, 1, 32, 32))

       #ic(x.shape)
        u = self.conv1(x)
        u = self.pool1(u)
       #ic(u.shape)
        u = self.conv2(u)
       #ic(u.shape)
        u = self.pool2(u)
       #ic(u.shape)
        u = u.reshape((bs,16,32,6,6))
       #ic(u.shape)
        u = self.expand(u)
        u = torch.nn.functional.relu(u)
       #ic("After Expanding")
       #ic(u.shape)
        sp = u.shape
        u = u.reshape((sp[0],sp[1] ,sp[2], sp[3] * sp[4]))
        # print(u.shape)
       #ic(u.grad)
        attented = []
       #ic(u.shape)
        u = u.reshape(16,bs,32,6,16)
        for i in range(16):
            q = u[i].reshape((32, bs, 96))
            v = u[i].reshape((32, bs, 96))
            k = u[i].reshape((32, bs, 96))
            att_out, att_weights = self.minimultiheads["M-atten-" + str(i)](q, k, v)
            attented.append(att_out)
       #ic(attented[0].shape

        u = torch.stack(attented)
        u = u.reshape((bs,16,32,96))
       #ic(u.shape)

        u = u.reshape(bs,16,3072)
        u = self.lin_bef_encod(u)
        u = nn.functional.relu(u)

       #ic(u.shape)
        u = u.reshape(16,bs,-1)
        u = self.encoder(u)
        # u = u.reshape((16,32,7))
        # print("Out")
       #ic(u.shape)
        u = u.reshape(bs,-1)
        u = self.full_con(u)
       #ic(u.shape)
        return u

