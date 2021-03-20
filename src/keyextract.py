import os
import cv2
import subprocess
import numpy as np
filename = "../dataset/1 blackboard_n/v_Blackboard_c1.mov"

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_i_keyframes(video_fn):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = basename+'_i_frame_'+str(frame_no)+'.jpg'
            cv2.imwrite(outname, frame)
            print ('Saved: '+outname)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

if __name__ == '__main__':
	video_path = filename
	p_frame_thresh = 5000
	cap = cv2.VideoCapture(video_path)
	ret, prev_frame = cap.read()
	while not ret:
		ret, prev_frame = cap.read()
	print("Cap is ",cap)
	prev_frame = cv2.resize(prev_frame, (128,128), 0, 0, cv2.INTER_CUBIC);
	prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY);
	processed_frames = []
	while ret:
		ret,curr_frame = cap.read()

		if ret:
			curr_frame = cv2.resize(curr_frame, (128,128), 0, 0, cv2.INTER_CUBIC);	
			curr_frame = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY);
			diff = cv2.absdiff(curr_frame,prev_frame)
			non_zero_count = np.count_nonzero(diff)
			if non_zero_count > p_frame_thresh:
				print('got frame')
				processed_frames.append(diff)
			prev_frame = curr_frame

print(len(processed_frames))

while 1:
	for p in processed_frames:	
		cv2.imshow('Frame',p)
				
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	


