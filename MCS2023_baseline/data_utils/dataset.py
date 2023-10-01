import os

import cv2
import pandas as pd
import torch.utils.data as data

from PIL import Image
import numpy as np
import torch

def read_image(image_file):
    img = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class Product10KDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = np.array(img)
        img = self.transforms(image=img)['image']
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = np.array(img)
        img = self.transforms(image=img)['image']
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __len__(self):
        return len(self.imlist)
