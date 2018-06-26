
import torch.nn as nn
from torchvision.datasets import SVHN
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim, save, load, from_numpy
import numpy as np

import cv2 as cv
import os

import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1   = nn.Linear(16*6*6, 128) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

from PIL import Image, ImageOps
def pad_img(img):
    desired_sz = 32
    w, h = img.size
    resize_ratio = desired_sz * 1.0 / max(w, h)
    img = img.resize((int(w*resize_ratio), int(h*resize_ratio)))
    
    #pad image to make 32x32
    w, h = img.size
    delta_w = desired_sz - w
    delta_h = desired_sz - h
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(
        img, 
        padding
    )

def proc_fn(x):
    return transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(x.convert('L')))

def preproc_pil_img(img):
    # gray, norm, tensor
    return proc_fn(pad_img(img))

def file2tensor(file_path):
    aa = preproc_pil_img(Image.open(file_path))
    c, w, h = aa.size()
    aa = aa.view((1, c, w, h))
    return aa

def cv2tensor(img):
    if len(img.shape) > 2:
        img = cv.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    aa = preproc_pil_img(pil_img)
    c, w, h = aa.size()
    aa = aa.view((1, c, w, h))
    return aa
    
def predict_image(tensor, model):
    output = model(Variable(tensor))
    return np.exp(output.data.numpy())
#     print(np.exp(output.data.numpy()))
#     return np.argmax(output.data.numpy(), axis=1)