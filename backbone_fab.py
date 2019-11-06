#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from unet import UNet
import random
import math
import torch


resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(channel, reduction, 1),
                nn.PReLU(),
                nn.Conv2d(reduction, channel, 1),
                nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
    def forward(self, x):
        y = self.conv(x)
        return y


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)



class BatchCrop(nn.Module):
    def __init__(self):
        super(BatchCrop, self).__init__()


    def forward(self, x):
        if self.training:
            flag = random.randint(0,1)
            if flag > 0:
                start = random.randint(0, 7)
                mask = x.new_ones(x.size())
                mask[:, :, start*2:(start+1)*2, :] = 0
                x = x * mask
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=1024, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        self.A_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.M_pool = torch.nn.AdaptiveMaxPool2d((1,1))

        add_block = []
        add_block += [nn.Linear(2048, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]
        add_block = nn.Sequential(*add_block)


        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.A_pool(x).squeeze() + self.M_pool(x).squeeze()
        x = self.add_block(x)
        f = x
        x = self.classifier(x)

        return f, x




class Network_D(nn.Module):
    def __init__(self, num_classes=751+10):
        super(Network_D, self).__init__()
        resnet18 = torchvision.models.resnet50()

        #self.register_buffer('lut', torch.zeros(
            #num_classes, 1024).cuda())

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = standard_create_layer(64, 64, 3, stride=1)
        #self.unet_1 = UNet(num_classes=256, in_channels=256, depth=4, merge_mode='concat')
        #self.bn_1 = nn.BatchNorm1d(256)
        #self.fc_1 = nn.Linear(in_features=256, out_features=1024, bias=True)

        self.layer2 = standard_create_layer(256, 128, 4, stride=2)
        #self.unet_2 = UNet(num_classes=512, in_channels=512, depth=4, merge_mode='concat')
        self.bn2 = nn.BatchNorm1d(512)
        self.att2 = SELayer(512,reduction=16)
        self.layer3 = standard_create_layer(512, 256, 6, stride=2)
        #self.layer3 = standard_create_layer(512, 256, 6, stride=2)
        self.att3 = SELayer(1024,reduction=16)
        self.layer4 = standard_create_layer(1024, 512, 3, stride=1)
        #self.unet_4 = UNet(2048, depth=4, merge_mode='concat')
        self.att4 = SELayer(2048,reduction=16)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(16,8), stride=1)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc_g = nn.Linear(in_features=2048, out_features=128, bias=True)
        

        self.bn3 = nn.BatchNorm1d(128)
        self.dp = nn.Dropout(0.5)
        
        self.fc_id_g = nn.Linear(in_features=128, out_features=num_classes, bias=True)

        # load pretrained weights and initialize added weight
        pretrained_state = model_zoo.load_url(resnet50_url)
        state_dict = self.state_dict()
        for k, v in pretrained_state.items():
            if 'fc' in k:
                continue
            state_dict.update({k: v})
        self.load_state_dict(state_dict)
        #nn.init.kaiming_normal_(self.fc.weight, a=1)
        #nn.init.constant_(self.fc.bias, 0)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # [B, 256, 56, 56]


        x = self.layer2(x) # [B, 512, 28, 28]

        att2 = self.att2(x)
        x = x + att2 * x

        x = self.layer3(x) # [B, 1024, 14, 14]
        att3 = self.att3(x)
        x = x + att3 * x
        

        x = self.layer4(x) # [B, 2048, 7, 7]
        att4 = self.att4(x)
        x =  x + x * att4




        x = F.max_pool2d(x, x.size()[2:]).view(x.size()[:2])
        x = self.bn4(x)
        x = self.dp(x)
        x = self.fc_g(x)
        #x = self.relu(x)
        #embd = self.bn3(x)
        embd = self.l2_norm(x)
        
        embd_p = embd
   
       
        if not self.training:

            #embd = self.l2_norm(embd_p)
            #embd_norm = torch.norm(embd_p, 2, 1, True).clamp(min=1e-12).expand_as(embd_p)
            #print(embd_norm.shape)
            #embd = embd_p / embd_norm
            return embd
        else:
            fc_id_g = self.fc_id_g(embd)
            #fc_id_p = self.fc_id_p(embd_p)

            return embd, fc_id_g
       


class Bottleneck(nn.Module):
    def __init__(self, in_chan, mid_chan, stride=1, stride_at_1x1=False, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)

        self.stride = stride

        
        self.mid_chan = mid_chan

        out_chan = 4 * mid_chan
        
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1,
                bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))
        #self.unet = UNet(num_classes=out_chan, in_channels=out_chan,depth=4, merge_mode='concat')

    def forward(self, x):
        x_in = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




def create_layer(in_chan, mid_chan, b_num, stride, flag=True):
    out_chan = mid_chan * 4
    blocks = [Pyramid_Bottleneck(in_chan, mid_chan, stride=stride, flag=flag),]
    for i in range(1, b_num):
        blocks.append(Pyramid_Bottleneck(out_chan, mid_chan, stride=1, flag=flag))
    return nn.Sequential(*blocks)

def standard_create_layer(in_chan, mid_chan, b_num, stride):
    out_chan = mid_chan * 4
    blocks = [Bottleneck(in_chan, mid_chan, stride=stride),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, mid_chan, stride=1))
    return nn.Sequential(*blocks)



if __name__ == '__main__':
    intensor = torch.randn(10, 3, 256, 128)
    net = Network_D()
    out = net(intensor)
    print(out.shape)

    params = list(net.parameters())
    optim = torch.optim.Adam(params, lr = 1e-3, weight_decay = 5e-4)
    lr = 3
    optim.defaults['lr'] = 4
    for param_group in optim.param_groups:
        param_group['lr'] = lr
        print(param_group.keys())
        print(param_group['lr'])
    print(optim.defaults['lr'])
    print(optim.defaults.keys())
    print(net)
