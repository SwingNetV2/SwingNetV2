import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # 첫번째 셀의 값을 빼서 [0,]로 초기값을 맞춰주는 역할

        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))
        # print(a)

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame

            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.transform: 
                      augmented = self.transform(image=img)  # img는 NumPy 배열
                      img = augmented["image"]
                    else:
                      img = img
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform: 
                      augmented = self.transform(image=img)  # img는 NumPy 배열
                      img = augmented["image"]
                else:
                  img = img
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        return sample
      
