#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:38:59 2023

@author: sazzadh
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import time
import config as cfg
from fns import *
import postprocessing
import pandas as pd
import architecture

data_dir = cfg.directory
test_transformer = transforms.Compose([transforms.ToTensor(),
                                  #transforms.RandomVerticalFlip(cfg.transform_variables['vertical_flip_probability']),
                                  #transforms.RandomHorizontalFlip(cfg.transform_variables['horizontal_flip_probability']),
                                  transforms.Normalize(mean=cfg.transform_variables['Normalize_mean'], std=cfg.transform_variables['Normalize_std'])
                                  ])

test_dataset = ImageFolder(data_dir + '/test', transform=test_transformer)

test_loader = DataLoader(test_dataset, shuffle = False, batch_size = cfg.test_batch_size)

myModel = architecture.CNN()
myModel.load_state_dict(torch.load('model.pth'))
test_Result = myModel.test(test_loader)
print(test_Result)