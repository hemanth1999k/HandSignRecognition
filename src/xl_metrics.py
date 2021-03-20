from  model import *
from attensat_xl import * 
import pickle 
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from run_1 import *
from icecream import *

choosen_model = 2
mindex = [5,8,10]
m_size  = mindex[choosen_model]



model = AttenModXL()
model.load_state_dict(torch.load("../models/divaten-X" + str(m_size)+"L-SD.pth",map_location=torch.device('cpu')))

videos = pickle.load(open("../saved/2FG-V.pkl","rb"))
labels = pickle.load(open("../saved/2FG-L.pkl","rb"))

images = [] 
images = []
frames_count = 0
imglabels = []
imglabels_direct = []
lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 

for i in range(len( videos)):
    v =  videos[i]
    l =  labels[i]
    for f in v:
        if np.count_nonzero(f) > 300:
            images.append(f)
            imglabels.append(lab_dic.index(l))
            imglabels_direct.append(l)
        frames_count+=1

images = np.array(images)[:,:,:]
imglabels = np.array(imglabels)
ic(images.shape)
ic(imglabels.shape)

X = torch.tensor(images)
Y = np.array(imglabels)

ic(X.shape)
ic(Y.shape)
model.eval()

# X,Y = it.X,it.Y
# pbar = tqdm.tqdm(range(100))


preds = []
preds_index = []
preds_args = []

inp_range = X.shape[0]
# inp_range=10
pbar = tqdm.tqdm(range(inp_range))
with torch.no_grad():
    for i in pbar:
        x = X[i].detach().clone()
        x = x.view(1,128,128)
        out = model(x)
        # preds_args.append(out.item())
        preds_args.append(torch.nn.functional.softmax(out).numpy())
        model.pop_memory(m_size)
        o_lab = torch.argmax(out,-1)
        preds.append(lab_dic[o_lab.item()])
        preds_index.append(o_lab.item())

# print(preds)
# print(Y[:100])

preds_args = np.array(preds_args).reshape((-1,25))
# y = imglabels[:100]
y = imglabels[:inp_range]
# n_values = np.max(y) + 1

n_values = 25 
y_true = np.eye(n_values)[y]
y_preds = np.eye(n_values)[preds_index]
ic(roc_auc_score(y_true,y_preds))


# print(preds)
# print(imglabels[:len(preds)])
# onehot1 = np.eye(n_values)[imglabels[:len(preds)]]

# print(onehot)
# print()
# print(onehot1)
# print(preds_args)

# print(onehot1.shape,preds_args.shape)
# ic(roc_auc_score(preds_args,onehot1[:,],multi_class="ovr"))
# ic(roc_auc_score(onehot1[:,],preds_args[:,],multi_class="ovr"))

    
# print(roc_auc_score(onehot1[:,0],preds_args[:0]))
# print(metrics.confusion_matrix(y,preds))
# print(metrics.classification_report(y,preds,digits=3))
 