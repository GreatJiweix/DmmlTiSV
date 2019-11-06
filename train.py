#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import logging
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from backbone_fab import Network_D

from balanced_sampler import BalancedSampler
from torch.autograd import Variable
from tqdm import tqdm
import random
from torch.utils.checkpoint import checkpoint_sequential
from losses import *
from vox_data import Vox_data
import torch.optim as optim




## logging
if not os.path.exists('./res/'): os.makedirs('./res/')
logfile = 'sphere_reid-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = os.path.join('res', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
num_per = 64
num_id = 4
data_dir = "./data/vox2_npy"
criterion = nn.CrossEntropyLoss()


def lr_scheduler(epoch, optimizer):
    warmup_epoch = 20
    warmup_lr = 1e-4
    lr_steps = [90, 230]
    start_lr = 1e-3
    lr_factor = 0.1

    if epoch <= warmup_epoch:  # lr warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
        lr = warmup_lr * (warmup_scale ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr
    else:  # lr jump
        for i, el in enumerate(lr_steps):
            if epoch == el:
                lr = start_lr * (lr_factor ** (i + 1))
                logger.info('====> LR is set to: {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.defaults['lr'] = lr
    lrs = [round(el['lr'], 6) for el in optimizer.param_groups]
    return optimizer, lrs

def margin_scheduler(epoch):
    warmup_epoch = 20
    warmup_lr = 1e-4
    lr_steps = [90, 200]
    start_lr = 1.0
    lr_factor = 0.1

    if epoch <= warmup_epoch:  # lr warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
        lr = warmup_lr * (warmup_scale ** epoch)
    else:  # lr jump
        for i, el in enumerate(lr_steps):
            if epoch == el:
                lr = start_lr * (lr_factor ** (i + 1))
    return lr






def load_network(network):
    save_path = "./res/model_32.pkl"


    network.load_state_dict(torch.load(save_path))
    return network


def train():
    ## data
    logger.info('creating dataloader')
    dataset = Vox_data(data_dir, is_train = True)
    num_classes = dataset.get_num_classes()
    print(num_classes)

    sampler = BalancedSampler(dataset, num_per, num_id)
    dl = DataLoader(dataset, batch_sampler = sampler, num_workers = 8)


    ## network and loss
    logger.info('setup model and loss')

    #nppair = NPairAngularLoss()
    npair = NPairLoss()
    angular = AngularLoss()
    tri = TripletLoss()

    net = Network_D(num_classes=num_classes)
    #net = PRNet(num_classes=num_classes)
    #net = load_network(net)
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    net = nn.DataParallel(net)
    net.train()
    net.cuda()
    net = load_network(net)


    

    ## optimizer
    logger.info('creating optimizer')
    params = list(net.parameters())
    optimizer = optim.Adam(params,lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #optimizer = optim.SGD(params, lr=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    #optim = torch.optim.Adam(params, lr = 1e-3)

    ## training
    logger.info('start training')
    t_start = time.time()
    loss_it = []
    top1_best = 0
    mAP_best = 0
    num_part = 6
    
    for ep in range(1000):
        
        net.train()
        #optim, lrs = lr_scheduler(ep, optim)
        #scheduler.step()
        

        for it, data in enumerate(dl):
            imgs = data[0]
            lbs = data[1]


            imgs = imgs.cuda()
            lbs = lbs.cuda()



            embd, fc_id = net(imgs)
            #print(embd.shape, fc_id.shape)
            #flag = random.randint(0,1)
            TL, prec = tri(embd, lbs)
            NPL = npair(embd, lbs)
            ANL = angular(embd, lbs) 
            #NA = nppair(embd, lbs)
            SM = criterion(fc_id, lbs)

            loss = 0.5*NPL + ANL + 0.1*SM + TL

            #if ep == 0 and it == 0:
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_it.append(loss.detach().cpu().numpy())
            if it % 10 == 0 and it != 0:
                _, predict = torch.max(fc_id.data, 1)
                total = lbs.size(0)
                correct = (np.array(predict.cpu()) == np.array(lbs.data.cpu())).sum()
                t_end = time.time()
                t_interval = t_end - t_start
                log_loss = sum(loss_it) / len(loss_it)
                msg = 'epoch: {}, iter: {}, loss: {:4f}, train_accuracy: {:.4f}, prec: {:.4f}, time: {:4f}'.format(ep,
                        it, log_loss, correct/total, prec, t_interval)
                #print(NPL, ANL, SM, TL)
                logger.info(msg)
                accuracy = correct/total
                if accuracy > top1_best:
                    top1_best = accuracy
                    torch.save(net.state_dict(), './res/model_best.pkl')

                loss_it = []
                t_start = t_end
        if ep % 10 == 0:
            torch.save(net.state_dict(), './res/model_{}.pkl'.format(ep)) 

                
            
        


    ## save model
    #torch.save(net.module.state_dict(), './res/model_final.pkl')
    logger.info('\nTraining done, model saved to {}\n\n'.format('./res/model_final.pkl'))


if __name__ == '__main__':
    train()
