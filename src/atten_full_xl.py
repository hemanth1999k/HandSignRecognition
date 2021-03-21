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

class Self_attention(nn.Module):
    def  __init__(self,dp = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dp)
        
    def forward(self,q,k,v,mask=None):
        matmul_qk = q @ k.transpose(-1,-2)
        scaled_attention_logits = matmul_qk/ torch.sqrt(torch.tensor(k.shape[-1],dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits,dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ v 
        return [output, attention_weights.detach().clone()]

class AttentionHead(nn.Module):
    def __init__(self,d_model,d_features,dp ):
        super().__init__()
        self.attn = Self_attention(dp)
        self.query_tfm = nn.Linear(d_model,d_features)
        self.key_tfm = nn.Linear(d_model,d_features)
        self.values_tfm = nn.Linear(d_model,d_features)
        self.kmemory = None 
        self.vmemory = None 
    
    def forward(self,queries,key,values,mask=None,discard_mem = False ):
        if discard_mem :
            self.kmemory = None 
            self.vmemory = None 
            
        Q = self.query_tfm(queries)
        K = self.key_tfm(key)
        V = self.values_tfm(values)
         
        dK = K.detach().clone()
        dV = V.detach().clone()

        if self.kmemory == None: 
            self.kmemory = dK 
            self.vmemory = dV 
        else:
            K = torch.cat((K,self.kmemory),dim=1)
            V = torch.cat((V,self.vmemory),dim=1)
             
            self.kmemory = torch.cat((dK,self.kmemory),dim=1)# concating in sequence length 
            self.vmemory = torch.cat((dV,self.vmemory),dim=1)
            # print("Memory appended",self.kmemory.shape)

        x,att_weight = self.attn(Q,K,V,None)
        return x,att_weight
    
    def pop_last(self,n):
        if self.kmemory.shape[1] == n*32:
            self.kmemory = self.kmemory[:,:-32,:]
            self.vmemory = self.vmemory[:,:-32,:]
    
    def print_seq_length(self):
        ic(self.kmemory.shape)
        ic(self.vmemory.shape)
        
class MultiheadAttentionXL(nn.Module):
    def __init__(self,d_model,d_feature,n_heads,dp=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        assert d_model == d_feature * n_heads
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dp) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
  
    def forward(self,queries,keys,values,discard_mem = False,mask=None):
        comb = [attn(queries, keys, values, mask=mask,discard_mem = discard_mem) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]


        # log_size(x[0], "output of single head")
        attentions = []
        xs = []
        for u,att in comb:
            xs.append(u)
            attentions.append(att)
        # reconcatenate
        x = torch.cat(xs, dim=-1) # (Batch, Seq, D_Feature * n_heads)
        attentions = torch.cat(attentions,dim=-1)
        # log_size(x, "concatenated output")
        x = self.projection(x) # (Batch, Seq, D_Model)
        # log_size(x, "projected output")
        return x,attentions

    def pop_last(self,n):
        for i,attn in enumerate(self.attn_heads):
            attn.pop_last(n)


class AttenModFullXL(torch.nn.Module):
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
        self.memory = []
        dicts = {}
        for i in range(16):
            dicts["M-atten-" + str(i)] = MultiheadAttentionXL(64,16, 4, dp=0.2)     
        
        self.multiheads = nn.ModuleDict(dicts)
        self.atten_dropout = nn.Dropout(0.2)
        
        self.layernorm1 = nn.LayerNorm(64)



        num_layers = 2
        self.xl_heads = {}
        self.norm_layers = {}
        for i in range(num_layers):
            self.xl_heads["XL-"+str(i)] = MultiheadAttentionXL(32*64,256,8,dp=0.2)
        self.xl_heads = nn.ModuleDict(self.xl_heads) 

        for i in range(num_layers):
            self.norm_layers["LN-"+str(i)] = nn.LayerNorm(32*64) 
        self.norm_layers = nn.ModuleDict(self.norm_layers)         

        
        self.flin1 = nn.Linear(16*2048,512)
        self.flin2 = nn.Linear(512,128)
        self.flin3 = nn.Linear(128,25)
        self.att_mat = None

    def pop_memory(self,n):
        for i in range(len(self.multiheads)):
            self.multiheads["M-atten-" + str(i)].pop_last(n)

        for i in range(len(self.xl_heads)):
            self.xl_heads["XL-"+str(i)].pop_last(n)

    def forward(self,t,discard_mem = False):
        t = t/255.0 # normaliztion added to the model forward itself
        bs = t.shape[0]
        sliced = torch.zeros((t.shape[0],16, 32, 32))
        # ic(x.shape)
        for i in range(4):
            for j in range(4):
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
        u = u.reshape((16,bs,32,64))
        attention_out =[]
        self.att_mat = []
        for i in range(len(self.multiheads)):
            att_out, _att_mat = (self.multiheads["M-atten-" + str(i)](u[i] ,u[i], u[i],discard_mem))
            self.att_mat.append(_att_mat.detach().clone())
            attention_out.append(att_out)
            
        with torch.no_grad():
            self.att_mat = torch.stack(self.att_mat)

        attention_out_stacked = torch.stack(attention_out) 
        
        u = u + self.atten_dropout(attention_out_stacked)
        u = self.layernorm1(u)
        u = u.reshape((16,bs,32*64))
        u = u.transpose(0,1) 

        for i in range(len(self.xl_heads.keys())):
            att_out,_att_mat = self.xl_heads["XL-"+str(i)](u,u,u,discard_mem)
            u = u+ torch.nn.functional.dropout(u)
            u = self.norm_layers["LN-"+str(i)]
        
        u = u.reshape((bs,16*2048))
        u = self.flin1(u)
        u = nn.functional.relu(u)
        u = self.flin2(u)
        u = nn.functional.relu(u)
        u = self.flin3(u)
        # ic(u.shape)
        return u


