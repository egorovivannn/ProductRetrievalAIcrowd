from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import torch
import pandas as pd
import numpy as np

import torch
import cv2


class Products_dataset():
    def __init__(self, df, data_path, 
                 img_dir='train', mode='train', 
                 transform=None
                 ):
        self.df = df
        self.data_path = data_path
        self.img_dir = img_dir
        self.augs = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # dataset_name = self.df.loc[idx, 'dataset'] if pd.notna(self.df.loc[idx, 'dataset'])   else ''
        # path = os.path.join(self.data_path, self.img_dir, dataset_name,  self.df.loc[idx, 'name'])
        path = self.df.loc[idx, 'name']
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # if self.df.loc[idx, 'dataset'] in ['consumer2shop', 'inshop']:
            #     # можно здесь на рандоме вырезать ббокс, типо аугментации
            #     h, w, _ = img.shape
            #     x1 = int(self.df.loc[idx, 'x_1']*w)
            #     y1 = int(self.df.loc[idx, 'y_1']*h)
            #     x2 = int(self.df.loc[idx, 'x_2']*w)
            #     y2 = int(self.df.loc[idx, 'y_2']*h)
            #     img = img[y1:y2, x1:x2]

            if self.augs:
                img = self.augs(image=img)['image']
                img = torch.from_numpy(img).permute(2, 0, 1)
        except Exception as e:
            print(e, path)
            img = torch.zeros((3, 600, 600))
        
        if self.mode=='test':
            return img
        elif self.mode=='temp':
            label = self.df.loc[idx, 'class']
            return img, label, path
        elif self.mode=='train' or 'val' in self.mode:
            label = self.df.loc[idx, 'class']
            return img, label
        return 