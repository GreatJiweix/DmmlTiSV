'''
import argparse
import scipy.io
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=770, type=int, help='test_image_index')
parser.add_argument('--test_dir',default="/mnt/mfs3/xujiwei/market1501/pytorch",type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

for i in range(1023):
    print(i)
    flag = False
    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result

    query_path, _ = image_datasets['query'].imgs[i]
    q_label = query_label[i]

    #print(query_path)
    #print('Top 10 images are as follow:')
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for j in range(10):
        ax = plt.subplot(1,11,j+2)
        ax.axis('off')
        #img_path, _ = image_datasets['gallery'].imgs[index[i]]
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]


        #label = gallery_label[index[i]]
        #imshow(img_path)
        #print(label, q_label)
        if label == q_label:
            #imshow(img_path)
            im = plt.imread(img_path)
            plt.imshow(im, cmap='Blues', interpolation='none', vmin=0, vmax=100, aspect='equal')
            ax.set_title('%d'%(j+1), color='green')
        else:
            if j == 0:
                print(query_path, img_path)
                flag = True
            imshow(img_path)
            ax.set_title('%d / wrong '%(j+1), color='red')
    if flag:
        fig.savefig("./figs/show_{}.png".format(i))

'''
import argparse
import scipy.io
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=770, type=int, help='test_image_index')
parser.add_argument('--test_dir',default="/mnt/mfs3/xujiwei/market1501/pytorch",type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    #print(qf.shape, ql.shape, qc.shape, gf.shape, gl.shape, gc.shape)
    temp = qf
    query = temp.view(-1,1)
    #print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


for i in range(1501):

    #i = opts.query_index
    #print(query_feature.shape,query_label.shape,query_cam.shape,gallery_feature.shape,gallery_label.shape,gallery_cam.shape)
    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result
    flag = False

    query_path, _ = image_datasets['query'].imgs[i]
    q_label = query_label[i]
    print(query_path)
    print('Top 10 images are as follow:')
    try: # Visualize Ranking Result 
    # Graphical User Interface is needed
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        for j in range(10):
            ax = plt.subplot(1,11,j+2)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'].imgs[index[j]]
            label = gallery_label[index[j]]
            imshow(img_path)
            if label == q_label:
                ax.set_title('%d'%(j+1), color='green')
            else:
                ax.set_title('%d'%(j+1), color='red')
                if j == 0:
                    flag = True
            print(img_path)
    except RuntimeError:
        for k in range(10):
            img_path = image_datasets.imgs[index[k]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    if flag:
        fig.savefig("./figs/show_{}.png".format(i))


