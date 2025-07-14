import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VideoDataSet(Dataset):

    def __init__(self, video_path: list, video_class: list, transform=None):
        self.video_path = video_path
        self.video_class = video_class
        self.transform = transform

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, item):
        cap = cv2.imread(self.video_path[item])
            
        label = self.video_class[item]
        img = preprocess_videodata(cap) 
            
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def preprocess_videodata(frame):
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = cv2.resize(frame, dsize=(640,640))
  frame = Image.fromarray(np.uint8(frame))
  frame = np.array(frame)
  frame = np.transpose(frame, (2, 0, 1))
  
  frame = torch.tensor(frame)
  frame = frame.float()
  frame = (frame - 127.5) / 127.5
  return frame