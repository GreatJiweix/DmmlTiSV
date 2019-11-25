from __future__ import absolute_import

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import torchvision
#from torch_deform_conv.layers import ConvOffset2D

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=2048, 
                    dropout=0.1, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained


        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        #change layer4[-1] relu to prelu
        self.base._modules['layer4'][-1].relu = nn.PReLU()
        #for i in self.base.parameters():
            #i.requires_grad=False

        self.num_features = num_features
        self.dropout = dropout
        self.num_classes = num_classes

        out_planes = self.base.fc.in_features
        # In deep person has_embedding is always False. So self.num_features==out_planes
        # for x2 embedding
        self.feat = nn.Linear(out_planes, self.num_features, bias=False)
        #for i in self.feat.parameters():
            #i.requires_grad=False
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.prelu = nn.PReLU()
        init.normal(self.feat.weight, std=0.001)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        #x2 classifier
        #self.classifier_x2 = nn.Linear(self.num_features, self.num_classes)
        self.features = nn.Linear(self.num_features, self.num_features)
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        #init.normal(self.features, std=0.001)
        #init.constant(self.features, 0)
        # x3 module
        self.att1 = SELayer(256,reduction=16)
        #for i in self.att1.parameters():
            #i.requires_grad=False
        self.att2 = SELayer(512,reduction=16)
        #for i in self.att2.parameters():
            #i.requires_grad=False
        self.att3 = SELayer(1024,reduction=16)
        #for i in self.att3.parameters():
            #i.requires_grad=False

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'layer2':
                att1 = self.att1(x)
                x = x + att1*x
                att1 = att1*x  
            if name == 'layer3':
                att2 = self.att2(x)
                x = x + att2*x
                att2 = att2*x
            if name == 'layer4':
                att3 = self.att3(x)
                x = x + att3*x
                att3 = att3*x
            if name == 'avgpool':
                break
            x = module(x)
        # triplet loss branch
        '''
        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        '''
        
        # global feature branch
        x2 = F.avg_pool2d(x, x.size()[2:])
        x2 = x2.view(x2.size(0), -1)
        x2 = self.feat(x2)
        x2 = self.feat_bn(x2)
        x2 = self.prelu(x2)
        x2 = self.drop(x2)
        x2 = self.features(x2)
        #x2 = x2.unsqueeze(2)
        #x2 = x2.unsqueeze(3)


        #features = self.l2_norm(x2)
        #features = 10 * features
        features = x2

        features = torch.squeeze(features)
        #features = self.l2_norm(x2)
        logits = self.classifier(features)

        





        # x3 module
        #######
        '''
        att1 = F.adaptive_avg_pool2d(att1,1)
        att1 = att1.view(att1.size(0),-1)
        att2 = F.adaptive_avg_pool2d(att2,1)
        att2 = att2.view(att2.size(0),-1)
        att3 = F.adaptive_avg_pool2d(att3,1)
        att3 = att3.view(att3.size(0),-1)
        att_feat = torch.cat([att1,att2,att3],1)
        att_feat = self.embed_x3(att_feat)
        att_feat = att_feat.unsqueeze(2)
        att_feat = att_feat.unsqueeze(3)
        att_feat = self.bn_x3(att_feat)
        att_feat = self.prelu(att_feat)
        att_feat = self.drop(att_feat)
        att_feat = torch.squeeze(att_feat)
        x3 = self.classifier_x3(att_feat)
        '''
        return logits, features

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

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

class Attention_module(nn.Module):
    def __init__(self,in_planes,feat_nums,num_classes):
        super(Attention_module, self).__init__()
        self.conv = nn.Conv2d(in_planes,feat_nums,3,padding=1,groups=1)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.classify_1 = nn.Linear(feat_nums,num_classes)
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
        feat = x = self.conv(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(x.size(0),-1)
        x = self.classify_1(x)
        pred = torch.max(x,1)[1]
        wt = self.classify_1.weight[pred.data] # B*C
        wt = wt.view(wt.size(0), wt.size(1),1,1)
        cam = (wt*feat).sum(1,keepdim=True)
        cam = fm_norm(cam)
        cam_mask = F.sigmoid(cam)
        return x,cam_mask

def fm_norm(inputs,p=2):
    '''
    input should have shape of B*C*H*W
    '''
    b,c,h,w = inputs.size()
    inputs = inputs.view(b, c, -1)
    inputs = F.normalize(inputs,p=p,dim=2)
    inputs = inputs.view(b,c,h,w)

    return inputs

def init_params(*modules):
    '''
    modules should be list or tuple
    '''
    def reset_params(m):
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, std=0.001)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=0.001)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight,1)
            init.constant(m.bias, 0)

    for m in modules:
        reset_params(m)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes,planes*4,1,bias=False),
                                         nn.BatchNorm2d(planes*4))
        #self.downsample = None
        self.stride = stride

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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

