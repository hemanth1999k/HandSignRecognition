import torch 
import pickle
import numpy as np
from icecream import ic
import cv2
import os 

def get_videos(data_loca):
    # for folder in os.listdir(data_loc):
    #     if folder[0] != '.':
    seen = []
    prev_tag = None
    not_worked =  0
    videos = []
    labels = []
    for file in os.listdir(data_loca):
        if file[0] == ".":
            continue
        print()
        cap = cv2.VideoCapture(data_loca + "/" + file)
        if "Copy" in file:
            name = file.split(" ")[2].split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            # ic(file)
        else:
            name = file.split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
        
        ic(tag,)
        frame_array = []
        
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(fcount):
            x,read_frame = cap.read()
            if x and i%2 == 0:
                frame_array.append(read_frame)
        

        captured_data = get_from_array(frame_array)
        ic(len(captured_data))
        videos.append(captured_data)
        labels.append(tag)
    return videos,labels
        
def get_from_array(video):
    w = 128
    frame = video[0]
    fcount = len(video) 
    pframe = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
    pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)

    all_frames = []
    removed = 0
    for i in range(1, fcount):
        frame = video[i]
        frame = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, pframe)
        diff[diff < 30] = 0
        count = np.count_nonzero(diff)
        
        if count > 10:
            all_frames.append(diff)
            # cv2.imshow('frame',diff)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        pframe = frame
    ic(removed)
    return all_frames

def convert_videos():
    dir = "/Volumes/Nantha/Workspace/Datasets/"
    folders = ["Dataset","Nanthak Dataset","Ram Dataset","tf_dataset","My Dataset"]
    videos = []
    labels = []
    for folder in folders:
        v, l = get_videos(dir+folder)
        print(len(v))
        ic("##########################")
        ic(len(v),len(l))

        videos.extend(v)
        labels.extend(l)
        ic(len(videos),len(labels))
        pickle.dump(videos,open("/Volumes/Nantha/Workspace/2FG-V.pkl","wb"))
        pickle.dump(labels,open("/Volumes/Nantha/Workspace/2FG-L.pkl","wb"))

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super(IterableDataset).__init__()
        ic("Loading videos")
        self.videos = pickle.load(open("/Volumes/Nantha/Workspace/2FG-V.pkl","rb")) 
        ic("Loading labels")
        self.labels = pickle.load(open("/Volumes/Nantha/Workspace/2FG-L.pkl","rb"))
        ic(set(self.labels))
        self.dic_labels = set(self.labels)
        self.maps = {}
        for lab in self.dic_labels:
            self.maps[lab] = []
        index = 0
        for lab in self.labels:
            self.maps[lab].append(index)
            index+=1

   def __iter__(self): 
       pass


if __name__ == '__main__':
    # convert_videos()
    it = IterableDataset()