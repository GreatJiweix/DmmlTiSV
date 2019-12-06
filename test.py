import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import io
import requests

from PIL import Image
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
#import cv2
import pdb
import torch
import torch.nn as nn
import glob
import random
import numpy as np
import time
import math
import sys
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
from config import Config
from sklearn import *
from sklearn import metrics as Metrics
from models import resnet
from torch.nn import DataParallel

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


def load_audio(filename, start=0, stop=None, resample=True):#
    sr = SAMPLE_RATE
    y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
    y = np.squeeze(y)
    return y, sr


class my_dataset(data.Dataset):
    def __init__(self, root, lists):
        self.root = root
        self.lists = lists

        self.transforms = transforms.Compose([
                transforms.ToTensor(),])

    def __getitem__(self, index):
        im_pth = self.lists[index]
        im_pth = os.path.join(self.root, im_pth)
        im, sr = load_audio(im_pth)
        S = librosa.core.stft(im, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#
        feature, _ = librosa.magphase(S)
        npy = np.log1p(feature)#
        npy = npy.transpose()
        npy = npy[np.newaxis, :, :]
        im = np.repeat(npy, 3, axis=0)
        #print(im.shape)



        num_frame = 300
        l = im.shape[1]
        ims = []
        for i in range(10):
            
            if l <= num_frame:
                new = np.zeros((3, num_frame, 161))
                new[:, :l, :] = im
                new[:, num_frame-l:, :] = im[:, :l, :]
                npy = new
            else:
                randint = np.random.randint(l - num_frame)
                npy = im[:, randint: randint+num_frame, :]
            #print(npy.shape)



            npy = np.swapaxes(npy,1,2)
            mu = np.average(npy)
            sigma = np.std(npy)
            npy = (npy - mu) / max(sigma, 0.001)
            #print(npy.shape)
            npy = self.transforms(npy).float()
            npy = npy.permute(1,2,0)
            npy = npy.unsqueeze(0)
            #print(npy.shape)
            if i == 0:
                ims = npy
            else:
                ims = torch.cat((ims, npy), 0)

        #print(ims.shape)


        return ims, self.lists[index]

    def __len__(self):
        return len(self.lists)






#root_dir = "/data2/xjw/vox1_test_npy/"
#test_path = "/home/xjw/workspace/metric_learning_speaker_verification/veri_test.txt"
#path = "/home/xjw/workspace/triplet-network-pytorch-master/veri_png.txt"

def get_lists(path):
    files= []
    f = open(path, "r")
    lines = f.readlines()
    for line in lines:
        line = line.split()
        #print(line[1], line[2])
        if line[1] not in files:
            files.append(line[1])
        if line[2] not in files:
            files.append(line[2])
    return files

def get_featurs(model, root_dir, lists):

    #pbar = tqdm(total=len(lists))
    #for idx, img_path in enumerate(lists):
    #    pbar.update(1)
    features_dict = {}

    trainloader = my_dataset(root=root_dir, lists=lists)
    dl = data.DataLoader(trainloader, batch_size=1)

    #trainloader = data.DataLoader(lists, batch_size=1)
    for i, (img, im_pth) in enumerate(dl):
        if i % 100 == 0:
            print(i)
        img = img.cuda()
        img = torch.squeeze(img)
        #img = img.unsqueeze(1)
        #print(i, img.shape, im_pth[0])
        if i == 0:
            _, feature = model(img)
            feature = feature.detach().cpu().numpy()
            features_dict[im_pth[0]] = feature 
            features = feature
        else:
            _, feature = model(img)
            feature = feature.detach().cpu().numpy()
            features_dict[im_pth[0]] = feature 
            features = np.concatenate((features, feature), axis=0)
    return features, features_dict




def cosin_metric(features1, features2):
    score = np.mean(np.matmul(features1, features2.T))
    #print("score:", score)
    return score



#score = score_vector
#label = target_label_vector


def calculate_eer_auc_ap(label,distance):

    fpr, tpr, thresholds = Metrics.roc_curve(label, distance, pos_label=1)
    AUC = Metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    AP = Metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x
    miss = 1 - fpr
    false = 1 - tpr
    #false = 1 - fpr
    #miss = 1 - tpr
    #C_det = 0.01*

    return EER,AUC,AP,fpr, tpr, miss, false

# K-fold validation for ROC



def vox_test(model, test_root, test_list):

    lists = get_lists(test_list)
    print(len(lists))

    f = open(test_list, "r")
    lines = f.readlines()
    score_vector = np.zeros((len(lines), 1))
    target_label_vector = np.zeros((len(lines), 1))
    features, features_dict = get_featurs(model, test_root, lists)

    lists = np.array(lists)

    scores = []
    labels = []

    for i, line in enumerate(lines):
        line = line.split()
        labels.append(int(line[0]))
        score = cosin_metric(features_dict[line[1]], features_dict[line[2]])
        #print(score)
        scores.append(score)    


    score = np.array(scores)[:, np.newaxis]
    label = np.array(labels)[:, np.newaxis]

#score_vector = softmax.softmax(score_vector)

#np.save(os.path.join('/home/xjw/workspace/metric_learning_speaker_verification/result/','score_vector.npy'),score)
#np.save(os.path.join('/home/xjw/workspace/metric_learning_speaker_verification/result/','target_label_vector.npy'),label)
    k = 1
    step = int(label.shape[0] / float(k))
    EER_VECTOR = np.zeros((k,1))
    AUC_VECTOR = np.zeros((k,1))
    C_det = np.zeros((k,1))
    for split_num in range(k):
        index_start = split_num * step
        index_end = (split_num + 1) * step
    #print(label[index_start:index_end])
    #print(score[index_start:index_end])
        EER_temp,AUC_temp,AP,fpr, tpr, miss, false = calculate_eer_auc_ap(label[index_start:index_end],score[index_start:index_end])
        EER_VECTOR[split_num] = EER_temp * 100
        AUC_VECTOR[split_num] = AUC_temp * 100
        C_det[split_num] = 10 *0.01* np.mean(miss) + 1 * 0.99 * np.mean(false)

    eer = np.mean(EER_VECTOR)
    print("eer:", eer)
    return eer





if __name__ == '__main__':

    opt = Config()
    #model = resnet34()


    logs_root = opt.logs_root
    logs = os.listdir(logs_root)

    model = resnet.resnet50(pretrained=False,num_classes=5994).cuda()
    model = nn.DataParallel(model)
    model.cuda()
    #model.load_state_dict(torch.load("./logs/model_avgpool2.pth.tar"))

    model.eval()
    vox_test(model, opt.vox_root, opt.vox_test_list)