from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
import cv2
# local modules
from torchvision import transforms as transforms
class MyCustomDataset(Dataset):
    def __init__(self, event_path, image_path, sequence_length, transform=transforms.ToTensor()):
        self.events_path = [os.path.join(event_path, e) for e in os.listdir(event_path)]
        self.events_path.sort()
        self.images_path = [os.path.join(image_path, i) for i in os.listdir(image_path)]
        self.images_path.sort()
        self.sequence_length = sequence_length
        self.transform=transform
        self.step_size = 50
        
        
        self.seq_len = (len(self.events_path)-self.sequence_length)//self.step_size+1

         
                  
    def __len__(self):
        indices = self.seq_len
        return indices
    
    def __getitem__(self, index):
        
        sequence = []
        x = [np.load(e) for e in (self.events_path[index*self.sequence_length:index*self.sequence_length+self.sequence_length])]
        x = [torch.from_numpy(e) for e in x]
        
        y = [cv2.imread(i) for i in (self.images_path[index*self.sequence_length:index*self.sequence_length+self.sequence_length])]
        y = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in y]
        y = [self.transform(i) for i in y]
        
        sequence = []

        for e in x:
            event = e
            for i in y:
                image = i

            item = {'events': event, 'images': image}
            sequence.append(item)
        return sequence
        
        

            
    
