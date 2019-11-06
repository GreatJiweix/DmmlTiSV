import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import io
import requests
from backbone_fab import Network_D
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
#import cosscore
#import softmax
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
#import folder
#import dataloader
MODEL_NAME = "./res/model_160.pkl"


net = Network_D(num_classes=5994)
#net = PRNet(num_classes=num_classes)
#net = load_network(net)
total = sum([param.nelement() for param in net.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
net = nn.DataParallel(net)
net.eval()
model = net.cuda()



model.load_state_dict(torch.load(MODEL_NAME))
use_gpu = torch.cuda.is_available()

use_gpu = torch.cuda.is_available()
print (" | CUDA available is {}".format(use_gpu))



class my_dataset(data.Dataset):
    def __init__(self, root, lists):
        self.root = root
        self.lists = lists

        self.transforms = transforms.Compose([
                transforms.ToTensor(),])

    def __getitem__(self, index):
        im_pth = self.lists[index]
        im_pth = os.path.join(self.root, im_pth)
        im_npy = np.load(im_pth.replace(".wav", ".npy"))

        im_npy = im_npy[np.newaxis, :, :]
        im_npy = np.repeat(im_npy, 3, axis=0)
        im_npy = im_npy.swapaxes(0, 2)
        im = self.transforms(im_npy).float()
        height = im.shape[2]
        ims = []

        for i in range(15):
            flag = random.randint(0, height-301)
            crop = im[:,:, flag:flag+300]
            crop = crop.unsqueeze(0)
            if i == 0:
                ims = crop
            else:
                ims = torch.cat((ims, crop), 0)

        return ims, self.lists[index]

    def __len__(self):
        return len(self.lists)






root_dir = "/home/xjw/data/vox1_test_npy/"
test_path = "/home/xjw/workspace/metric_learning_speaker_verification/veri_test.txt"
#path = "/home/xjw/workspace/triplet-network-pytorch-master/veri_png.txt"

def get_lits(path):
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

lists = get_lits(test_path)
print(len(lists))


f = open(test_path, "r")
lines = f.readlines()
score_vector = np.zeros((len(lines), 1))
target_label_vector = np.zeros((len(lines), 1))




def get_featurs(root_dir, lists):

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
        #print(i, img.shape, im_pth[0])
        if i == 0:
            feature = model(img)
            feature = feature.detach().cpu().numpy()
            features_dict[im_pth[0]] = feature 
            features = feature
        else:
            feature = model(img)
            feature = feature.detach().cpu().numpy()
            features_dict[im_pth[0]] = feature 
            features = np.concatenate((features, feature), axis=0)
    return features, features_dict

features, features_dict = get_featurs(root_dir, lists)
#print(features.shape)

#def get_feature_dict(lists, features):
#    fe_dict = {}
#    for i, each in enumerate(lists):
#        if i < 100:
#            fe_dict[each] = features[i:i+10, :]
        #print(fe_dict[each].shape)
#    return fe_dict

#features_dict = get_feature_dict(lists, features)

def cosin_metric(features1, features2):
    score = np.mean(np.matmul(features1, features2.T))
    #print("score:", score)
    return score



lists = np.array(lists)

scores = []
labels = []

for i, line in enumerate(lines):
    line = line.split()
    labels.append(int(line[0]))
    score = cosin_metric(features_dict[line[1]], features_dict[line[2]])
    scores.append(score)    


score = np.array(scores)[:, np.newaxis]
label = np.array(labels)[:, np.newaxis]

#score_vector = softmax.softmax(score_vector)

np.save(os.path.join('/home/xjw/workspace/metric_learning_speaker_verification/result/','score_vector.npy'),score)
np.save(os.path.join('/home/xjw/workspace/metric_learning_speaker_verification/result/','target_label_vector.npy'),label)



#score = score_vector
#label = target_label_vector


def calculate_eer_auc_ap(label,distance):

    fpr, tpr, thresholds = metrics.roc_curve(label, distance, pos_label=1)
    AUC = metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    AP = metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

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
print("EER=",np.mean(EER_VECTOR))
print("AUC=",np.mean(AUC_VECTOR))
print("C_det=",np.mean(C_det))







'''
import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import torch
import torch.nn as nn
import glob

MODEL_NAME = './model_avgpool.pth.tar'
if os.path.exists(MODEL_NAME):
    model_weights = torch.load(MODEL_NAME)
    print("load tar")
else:
    raise IOError

model = models.resnet18(pretrained=True)

class novelmodel(nn.Module):
    def __init__(self):
        super(novelmodel, self).__init__()
    #    self.conv0 = nn.Conv2d(3, 64, 5, 2, 2, bias=False)
        self.features = nn.Sequential(
            *list(model.children())[:-2]
        )
        self.conv1 = torch.nn.Conv2d(512, 1211, kernel_size=(1, 1), stride=1)
        #self.dropout = torch.nn.Dropout(0.05)
        self.avgpool1 = torch.nn.AvgPool2d((4,1))

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.conv1(x))
        #x = self.dropout(x)
        #print(x.shape)
        
        x = self.avgpool1(x)
        x = torch.squeeze(x)
        out = x
        


        #print(x.size())
        return x, out


model = novelmodel()

model.load_state_dict(model_weights)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = '/data2/xjw/vox1_kaldi_resnet18/test/'
checkpoint_path = '/home/xjw/workspace/resnet_sv/result/en/'

n_classes = 1211
batch_size = 64
torch.cuda.set_device(0)
use_gpu = torch.cuda.is_available()
print (" | CUDA available is {}".format(use_gpu))

def read_utt(path):
    us_en = []
    ss_en = []
    dir_firsts = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    dir_firsts.sort()
    for idx, folder in enumerate(dir_firsts):
        npys = []
        for utt in glob.glob(folder + '/*.png'):
            npys.append(utt)
        npys.sort()
        for i in range(len(npys)):
            if i < 5:
                us_en.append(npys[i])
                ss_en.append(idx)                                
            else:
                pass
    return np.array(us_en), np.array(ss_en)

x_en, y_en = read_utt(path)


# transforms
normalize = transforms.Normalize(
   mean=[0.5, 0.5, 0.5],
   std=[0.6, 0.6, 0.6]
)
preprocess = transforms.Compose([
   #transforms.CenterCrop((20,100)),
   transforms.ToTensor(),
   normalize
])

data_transform = transforms.Compose([
        transforms.CenterCrop((128,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.6, 0.6, 0.6])
    ])



#print(weight_softmax.shape)

#print(features.shape)
        ################################################
        ############## ENROLLMENT Model ################
        ################################################


    # The model predefinition.
NumClasses = 40
NumLogits = 1211
#MODEL = np.zeros((NumClasses, NumLogits, 4), dtype=np.float32)
MODEL = np.zeros((NumClasses, NumLogits), dtype=np.float32)

    # Go through the speakers.
model.eval()
for speaker_id, speaker_class in enumerate(range(0, 40)):

        # Get the indexes for each speaker in the enrollment data
    speaker_index = np.where(y_en[:] == speaker_class)[0]
    
        # Enrollment of the speaker with specific number of utterances.
        #speaker_enrollment, label_enrollment = x_en[start_idx:end_idx, :, :], y_en[start_idx:end_idx]
    speaker_enrollment, label_enrollment = x_en[speaker_index], y_en[speaker_index]


    
    #for i in range(len(speaker_enrollment)):

    #feature_final = torch.ones((len(speaker_enrollment)))
    temps = torch.ones((speaker_enrollment.shape[0],3,128,32))
    for i in range(speaker_enrollment.shape[0]):
        temp = speaker_enrollment[i]
        temp = data_transform(Image.open(temp))
        temps[i,:,:,:] = temp
    inputs = temps





    #inputs = data_transform(Image.open(speaker_enrollment[0]))
    #print(inputs.shape)

    inputs = Variable(inputs.cuda())
    #inputs = inputs.expand(1,3,300,20)

    _, feature = model(inputs)
    #print(feature.shape)
    feature = feature.data.cpu().numpy()
    feature = np.squeeze(feature)
    feature =  np.mean(feature, axis=0)
    #print(feature.shape)



            # # # L2-norm along each utterance vector
            # feature_speaker = sklearn.preprocessing.normalize(feature_speaker,norm='l2', axis=1, copy=True, return_norm=False)

            # Averaging for creation of the spekear model
    speaker_model = feature

            # Creating the speaker model
    MODEL[speaker_id,:] = speaker_model
        #print(speaker_id)

        #if not os.path.exists(FLAGS.enrollment_dir):
            #os.makedirs(FLAGS.enrollment_dir)
        # Save the created model.
    np.save(os.path.join(checkpoint_path , 'MODEL.npy'), MODEL)


'''


'''
import numpy as np
import os
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = '/data2/xjw/vox1_mfcc_test_60/'
checkpoint_path = '/home/xjw/workspace/dnn_sv/result_resnet/en/'

n_classes = 1211
batch_size = 64
def read_utt(path):
    us_en = []
    ss_en = []
    dir_firsts = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    dir_firsts.sort()
    for idx, folder in enumerate(dir_firsts):
        npys = []
        for utt in glob.glob(folder + '/*.npy'):
            npys.append(utt)
        npys.sort()
        for i in range(len(npys)):
            if i < 5:
                utt = np.load(npys[i])
                if utt.shape[0]>=300:
                    for k in range(int(utt.shape[0]/300)):
                        us_en = list(us_en)
                        ss_en = list(ss_en)
                        us_en.append(utt[300*k:300*(k+1),:])
                        ss_en.append(idx)
                        us_en = list(us_en)
                        ss_en = list(ss_en)
                    
                else:
                    n = utt.shape[0]
                    temp = np.zeros((300, 60))
                    temp[:n, :] = utt
                    temp[n:300, :] = utt[:300-n, :]
                        #temp = np.reshape(temp, (100,100))
                    us_en = list(us_en)
                    ss_en = list(ss_en)
                    us_en.append(temp)
                    ss_en.append(idx)
                    us_en = list(us_en)
                    ss_en = list(ss_en)
                                  
            else:
                pass
        us_en = np.array(us_en)
        #us_en = us_en[:, :, :, np.newaxis]
        ss_en = np.array(ss_en)
    return np.array(us_en), np.array(ss_en)
x_en, y_en = read_utt(path)
print(x_en.shape, y_en.shape)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]




x = tf.placeholder(tf.float32,shape=[None,300,60,1],name='x')
y_ = tf.placeholder(tf.int32,shape=[None,],name='y_')
y2 = tf.one_hot(y_,n_classes)


#x = x[:, :, :, np.newaxis]
net, end_points = nets.resnet_v1.resnet_v1_50(x, num_classes=n_classes,is_training=True)
net = tf.squeeze(net, axis=[1, 2])
logits = slim.fully_connected(net, num_outputs=n_classes,activation_fn=None, scope='Predict')

print logits.shape 
#
#for key in end_points:
    #print(key, end_points[key].shape)

features = end_points['predictions']
features = tf.squeeze(features, axis=[1, 2])
print(features.shape)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore, max_to_keep=20)
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

        ################################################
        ############## ENROLLMENT Model ################
        ################################################

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir='/home/xjw/workspace/dnn_sv/result_resnet_50')
    saver.restore(sess, latest_checkpoint)

    # The model predefinition.
    NumClasses = 40
    NumLogits = 1211
    MODEL = np.zeros((NumClasses, NumLogits), dtype=np.float32)

    # Go through the speakers.
    for speaker_id, speaker_class in enumerate(range(0, 40)):
        #print(speaker_id, speaker_class)

        # Get the indexes for each speaker in the enrollment data
        speaker_index = np.where(y_en[:] == speaker_class)[0]

        # Enrollment of the speaker with specific number of utterances.
        #speaker_enrollment, label_enrollment = x_en[start_idx:end_idx, :, :], y_en[start_idx:end_idx]
        speaker_enrollment, label_enrollment = x_en[speaker_index, :, :], y_en[speaker_index]

        speaker_enrollment = speaker_enrollment[:, :, :, np.newaxis]

        # Evaluation
        feature = sess.run(features,feed_dict={x: speaker_enrollment, y_: label_enrollment})

        #Extracting the associated numpy array.
        #feature_speaker = feature[0]
        feature_speaker = np.zeros(NumLogits, dtype=np.float32)
        for i in range(feature.shape[0]):
            feature_speaker += feature[i]
        feature_speaker = feature_speaker / feature.shape[0]

            # # # L2-norm along each utterance vector
            # feature_speaker = sklearn.preprocessing.normalize(feature_speaker,norm='l2', axis=1, copy=True, return_norm=False)

            # Averaging for creation of the spekear model
        speaker_model = feature_speaker

            # Creating the speaker model
        MODEL[speaker_id,:] = speaker_model
        #print(speaker_id)

        #if not os.path.exists(FLAGS.enrollment_dir):
            #os.makedirs(FLAGS.enrollment_dir)
        # Save the created model.
        np.save(os.path.join(checkpoint_path , 'MODEL.npy'), MODEL)











'''
'''
class novelmodel(nn.Module):
    def __init__(self):
        super(novelmodel, self).__init__()
    #    self.conv0 = nn.Conv2d(3, 64, 5, 2, 2, bias=False)
        self.features = nn.Sequential(
            *list(model.children())[:-2]
        )
        self.bn1 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)     
        self.conv1 = torch.nn.Conv2d(2048, 4096, kernel_size=(1, 1), stride=1)
        self.bn2 = torch.nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True)
        self.avgpool1 = torch.nn.AvgPool2d((7,7))
        self.conv2 = torch.nn.Conv2d(4096, 1024, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
        self.conv3 = torch.nn.Conv2d(1024, 1211, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        #out = x
        x = self.bn2(x)
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = torch.squeeze(x)
        x = self.bn3(x)
        
        x = self.conv3(x)
        x = torch.squeeze(x)
        #print(x.size())
        return x, x

'''
