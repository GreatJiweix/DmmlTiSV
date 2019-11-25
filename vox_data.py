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

#from random_erasing import RandomErasing
import random
import glob
import os
import audio_processing
#import vad
from scipy.io import wavfile
import librosa
from scipy.signal.windows import hamming
import soundfile as sf

SAMPLE_RATE = 16000
FEATURE = 'fft'#
FEATURE_LEN = 161#
WIN_LEN = 0.02#
WIN_STEP = 0.01#

N_FFT = int(WIN_LEN * SAMPLE_RATE)#
HOP_LEN = int(WIN_STEP * SAMPLE_RATE)#

N_FRAMES = 300#
DURATION = (N_FRAMES - 1) * WIN_STEP#
N_SAMPLES = int(DURATION * SAMPLE_RATE)#

N_TEST_FRAMES = 300#
TEST_DURATION = (N_TEST_FRAMES - 1) * WIN_STEP#
N_TEST_SAMPLES = int(TEST_DURATION * SAMPLE_RATE)#

noise_dir = "/home/xjw/data/RIRS_NOISES/pointsource_noises/"
noise_files = "/home/xjw/data/RIRS_NOISES/noise.scp"


def get_one_noisefile(noise_file=noise_files):
    f = open(noise_file, 'r')
    noise_files = f.readlines()
    num_noises = len(noise_files)
    num = random.randint(0, num_noises-1)
    noise_file = noise_files[num].split()[0]
    return noise_file


def load_audio(filename, start=0, stop=None, resample=True):#
    sr = SAMPLE_RATE
    y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
    y = np.squeeze(y)
    return y, sr

def add_noise(audio_path, noise_path, percent=0.5, sr=16000):
    src, sr = librosa.load(audio_path, sr=sr)
    src_noise, sr = librosa.load(noise_path, sr=sr)
    #print(len(src), len(src_noise))
    if len(src) > len(src_noise):
        n = int(len(src)/len(src_noise))
        src_noise = src_noise.repeat(n+1)
    flag = random.randint(0, len(src_noise) - len(src))
    src_noise = src_noise[flag: flag+len(src)]
    percent = 0.002*random.randint(1,5)
    src = src + percent * src_noise
    S = librosa.core.stft(src, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#
    feature, _ = librosa.magphase(S)
    return feature


class Vox_data(Dataset):
    def __init__(self, data_pth, is_train = True, *args, **kwargs):
        super(Vox_data, self).__init__(*args, **kwargs)

        ## parse image names to generate image ids
        imgs = []
        labels = []
        dir_names = os.listdir(data_pth)
        for i, dir in enumerate(dir_names):
            ims = os.path.join(data_pth, dir)
            for im in glob.glob(ims + '/*.wav'):
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
                #transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
                #RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])
        ## H-Flip
        self.trans_no_train_flip = transforms.Compose([
                #transforms.Resize((256,128), interpolation=3),
                #transforms.Resize((256, 128)),
                #transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
            ])
        self.trans_no_train_noflip = transforms.Compose([
                #transforms.Resize((256,128), interpolation=3),
                #transforms.Resize((256, 128)),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
            ])


    def __getitem__(self, idx):
        im_pth = self.im_pths[idx]
        pid = self.labels[idx]
        #im = Image.open(im_pth)

        num_frame = 300
        noise_file = get_one_noisefile(noise_files)
        noise_file = noise_dir + noise_file
        feature = add_noise(im_pth, noise_file)
        npy = np.log1p(feature)#
        npy = npy.transpose()
        npy = npy[np.newaxis, :, :]
        npy = np.repeat(npy, 3, axis=0)

        l = npy.shape[1]
        if l <= num_frame:
            new = np.zeros((3, num_frame, 161))
            new[:, :l, :] = npy
            new[:, num_frame-l:, :] = npy[:, :l, :]
            npy = new
        else:
            randint = np.random.randint(l - num_frame)
            npy = npy[:, randint: randint+num_frame, :]
        npy = np.swapaxes(npy,1,2)
        mu = np.average(npy)
        sigma = np.std(npy)
        npy = (npy - mu) / max(sigma, 0.001) 


        #print(im_npy.shape)
        



        im = self.trans_train(npy)

        im = im.permute(1,2,0)


        #print(im.shape)
            


        return im.float(), self.pid_label_map[pid], self.im_infos[im_pth], self.im_infos[im_pth]
        #return im.float(), pid

    def __len__(self):
        return len(self.im_pths)

    def get_num_classes(self):
        return len(list(self.person_infos.keys()))

if __name__ == "__main__":
    ds_train = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    ds_test = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_test', is_train = False)
    im, lb, _ = ds_train[10]
