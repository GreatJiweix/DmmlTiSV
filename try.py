from PIL import Image
import numpy as np 
import math
import shutil
import random
import torch

img_pth = "/home/xujiwei/dataset/MSMT17_V2/cat.jpg"



def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

a, b, c, d = torch.rand(1,100), torch.rand(1,100), torch.rand(1,100), torch.rand(1,100)

a, b, c, d = l2_norm(a), l2_norm(b), l2_norm(c), l2_norm(d)

A, B, C, D = a, b, c, d

dist_ab, dist_cd = torch.dist(a, b, p=2), torch.dist(c, d, p=2)
dist_AB, dist_CD = torch.matmul(A, B.t()), torch.matmul(C, D.t())
print(dist_ab - dist_cd)
print(dist_AB - dist_CD)
