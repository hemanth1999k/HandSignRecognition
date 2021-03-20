
#divaten_model_colab1
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
        sliced = torch.zeros((t.shape[0],16, 32, 32)).cuda()
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
        sliced = torch.zeros((t.shape[0],16, 32, 32)).cuda()
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
