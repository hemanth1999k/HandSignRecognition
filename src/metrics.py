from  model import *
import pickle 
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

processed_videos = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/allvideos.pkl","rb"))
labels = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/alllabels.pkl","rb"))


def transformY(y):
    av = list(set(labels))
    keys_values = str(av)

    with open("label_index.txt","w") as f:
        f.write(keys_values)
        f.close()
    Y = []
    for x in y:
        Y.append(av.index(x))
    return np.array(Y)

Y = transformY(labels)

all = np.array(processed_videos)
all = all[:,np.newaxis,:,:,:]
model = Model(25)
model.load_state_dict(torch.load("../models/model_6.pth"))
model.eval()
dic_ = ['classroom', 'title', 'sentence', 'pencil', 'college', 'eraser', 'picture', 'fail', 'book', 'word', 'exam', 'principal', 'blackboard', 'university', 'paper', 'memorize', 'teacher', 'file', 'education', 'pass', 'pen', 'student', 'result', 'scale', 'name']
#dic_ = ['pencil', 'paper', 'pass', 'sentence', 'result', 'blackboard', 'memorize', 'education', 'pen', 'exam', 'college', 'principal', 'scale', 'book', 'fail', 'name', 'eraser', 'file', 'university', 'title', 'teacher', 'student', 'word', 'classroom', 'picture']
predicted = []
predicted_numeric = []
count = 0


for x in all: 
    count+=1
    if count %100 == 0:
        print(count)
    u = x[np.newaxis,:,:,:]
    u = torch.tensor(u,dtype=torch.float32)
    k = torch.argmax(model(u))
    predicted.append(dic_[k])
    predicted_numeric.append(k)

    if count == 400:
        break

# print(predicted)
# print(labels)
print(metrics.confusion_matrix(labels[:len(predicted)], predicted))
print(metrics.classification_report(labels[:len(predicted)], predicted, digits=3))


print(Y[:len(predicted)])
n_values = np.max(Y) + 1

onehot = np.eye(n_values)[Y[:len(predicted)]]
onehot1 = np.eye(n_values)[predicted_numeric]
print(roc_auc_score(onehot,onehot1,multi_class='ovr'))

# print(all.shape)
