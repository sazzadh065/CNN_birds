# -*- coding: utf-8 -*-
"""

@author: Sazzad
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
classes = os.listdir(data_dir + "/train")

augment_transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomVerticalFlip(cfg.transform_variables['vertical_flip_probability']),
                                  transforms.RandomHorizontalFlip(cfg.transform_variables['horizontal_flip_probability']),
                                  transforms.Normalize(mean=cfg.transform_variables['Normalize_mean'], std=cfg.transform_variables['Normalize_std'])
                                  ])

valid_transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomVerticalFlip(cfg.transform_variables['vertical_flip_probability']),
                                  transforms.RandomHorizontalFlip(cfg.transform_variables['horizontal_flip_probability']),
                                  transforms.Normalize(mean=cfg.transform_variables['Normalize_mean'], std=cfg.transform_variables['Normalize_std'])
                                  ])

normalize_transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=cfg.transform_variables['Normalize_mean'], std=cfg.transform_variables['Normalize_std'])
                                  ])


train_dataset = torch.utils.data.ConcatDataset([ImageFolder(data_dir + '/train', transform=augment_transformer), 
                                                ImageFolder(data_dir + '/train', transform=normalize_transformer)])


valid_dataset = torch.utils.data.ConcatDataset([ImageFolder(data_dir + '/valid', transform=valid_transformer), 
                                                ImageFolder(data_dir + '/valid', transform=normalize_transformer)])

batch_size = cfg.batch_size
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
valid_loader = DataLoader(valid_dataset, shuffle = True, batch_size = batch_size)


#training

myModel = architecture.CNN()
history = myModel.fit(cfg.epoch,cfg.learning_rate,cfg.weight_decay, train_loader, valid_loader)

postprocessing.plot_accuracies(history)
#plt.savefig('accuracy.jpg')

postprocessing.plot_losses(history)
#plt.savefig('losses')
#print(history)

#train_losses = [x['train_loss'] for x in history]
#val_losses = [x['val_loss'] for x in history]
#accuracies = [x['val_acc'] for x in history]
        
torch.save(myModel.state_dict(), 'model.pth')
#history_df = pd.DataFrame.from_dict(history)
#history_df.to_csv('history.csv')