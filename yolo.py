from __future__ import division
import time
from util import *
import os
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools
import torch



def new_input():
    img = cv2.imread("test.jpg")
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    return img_




image = 'test.jpg'
batch_size = 1
confidence = .25
nms_thresh = .25
start = 0

num_classes = 52

print("loading classes and weights...")
classes = load_classes('aces.names')

model = Darknet('aces.cfg')
model.load_weights('aces_4000.weights')
print("model loaded...")

model.net_info["height"] = 16
inp_dim = int(model.net_info["height"])





model.eval()

write = False
model(new_input(), False)



objs = {}

