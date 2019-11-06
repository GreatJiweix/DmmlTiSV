#! /usr/bin/python
# -*- encoding: utf-8 -*-


import os
import os.path as osp
import numpy as np
#import cv2
import random
import math
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from random_erasing import RandomErasing
import random
import glob
import os
#from detect import detect
#import transforms


class Vox_data(Dataset):
    def __init__(self, data_pth, is_train = True, *args, **kwargs):
        super(Vox_data, self).__init__(*args, **kwargs)

        ## parse image names to generate image ids
        imgs = []
        labels = []
        dir_names = os.listdir(data_pth)
        for i, dir in enumerate(dir_names):
            ims = os.path.join(data_pth, dir)
            for im in glob.glob(ims + '/*.npy'):
                imgs.append(im)
                labels.append(i)

        imgs, labels = np.array(imgs), np.array(labels)

        self.is_train = is_train
        self.im_pths = imgs
        self.im_infos = {}
        self.person_infos = {}
        self.labels = labels


        for i, im in enumerate(imgs):
            im_pth = self.im_pths[i]
            pid = int(labels[i])


            self.im_infos.update({im_pth: pid})
            if pid in self.person_infos.keys():
                self.person_infos[pid].append(i)
            else:
                self.person_infos[pid] = [i, ]

        self.pid_label_map = {}
        for i, (pid, ids) in enumerate(self.person_infos.items()):
            self.person_infos[pid] = np.array(ids, dtype = np.int32)
            self.pid_label_map[pid] = i

        ## preprocessing

        self.trans_train = transforms.Compose([
                #transforms.Resize((112,112), interpolation=3),
                #transforms.Pad(5),
                #transforms.RandomCrop((112, 112)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])
        ## H-Flip
        self.trans_no_train_flip = transforms.Compose([
                #transforms.Resize((256,128), interpolation=3),
                #transforms.Resize((256, 128)),
                #transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.trans_no_train_noflip = transforms.Compose([
                #transforms.Resize((256,128), interpolation=3),
                #transforms.Resize((256, 128)),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


    def __getitem__(self, idx):
        im_pth = self.im_pths[idx]
        pid = self.labels[idx]
        #im = Image.open(im_pth)

        im_npy = np.load(im_pth)

        im_npy = im_npy[np.newaxis, :, :]
        im_npy = np.repeat(im_npy, 3, axis=0)
        im_npy = im_npy.swapaxes(0, 2)



        #print(im_npy.shape)
        

        if self.is_train:

            im = self.trans_train(im_npy)
            
        else:
            im_noflip = self.trans_no_train_noflip(im)
            im_flip = self.trans_no_train_flip(im)
            im = [im_noflip, im_flip]
        #print(im_pth, pid, im.shape)
        #print(im.shape)
        height = im.shape[2]
        if height < 300:
            pass
        else:
            flag = random.randint(0, height-301)
            im = im[:,:, flag:flag+300 ]
        #print("im.shape:",im.shape) 

        return im.float(), self.pid_label_map[pid], self.im_infos[im_pth], self.im_infos[im_pth]

    def __len__(self):
        return len(self.im_pths)

    def get_num_classes(self):
        return len(list(self.person_infos.keys()))

if __name__ == "__main__":
    ds_train = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    ds_test = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_test', is_train = False)
    im, lb, _ = ds_train[10]
