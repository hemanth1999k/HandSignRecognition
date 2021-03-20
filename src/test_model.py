import time
from icecream import ic
from model import * 
import torch
import cv2 
import pickle

model = Model(25) 

model1 = Model(25) 

model2 = Model(25) 

model.load_state_dict(torch.load("../models/model_1.pth"))

model1.load_state_dict(torch.load("../models/model_2.pth"))

model2.load_state_dict(torch.load("../models/model_3.pth"))

processed_videos = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/allvideos.pkl","rb"))
alllabels = pickle.load(open("/Volumes/Nantha/Workspace/gesres/saved/alllabels.pkl","rb"))


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

Y = transformY(alllabels)

all = np.array(processed_videos)
all = all[:,np.newaxis,:,:,:]



model.eval()
model1.eval()
model2.eval()

cap = cv2.VideoCapture(0)
x,f = cap.read()

while not x:
    x,f = cap.read()
st = time.time() 
frames = [f]


while time.time()- st < 3:
    print(time.time() - st)
    x,f = cap.read()
    frames.append(f)


ic(np.array(frames).shape)
input = get_from_array(frames)
input = input[np.newaxis,np.newaxis,:,:,:]

ic(input.shape)
input = torch.tensor(input, dtype = torch.float32)

y = model(input)
y1 = model1(input)
y2 = model2(input)

onehot = torch.argmax(y)
onehot1 = torch.argmax(y1)
onehot2 = torch.argmax(y2)
#print(onehot)

dic = ['classroom', 'title', 'sentence', 'pencil', 'college', 'eraser', 'picture', 'fail', 'book', 'word', 'exam', 'principal', 'blackboard', 'university', 'paper', 'memorize', 'teacher', 'file', 'education', 'pass', 'pen', 'student', 'result', 'scale', 'name']     

print(dic[onehot],dic[onehot1],dic[onehot2])
input = input.numpy()[0][0]
for i in input:
    cv2.imshow("frame",i)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


