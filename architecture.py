#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:34:23 2023

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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(cfg.input_channel, cfg.number_of_filters[0], kernel_size = cfg.filter_size[0], padding = cfg.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(cfg.maxpool_filter_size,cfg.maxpool_stride),
            
            nn.Conv2d(cfg.number_of_filters[0],cfg.number_of_filters[1], kernel_size = cfg.filter_size[1], stride = cfg.conv_stride, padding = cfg.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(cfg.maxpool_filter_size,cfg.maxpool_stride),
            
            nn.Conv2d(cfg.number_of_filters[1], cfg.number_of_filters[2], kernel_size = cfg.filter_size[2], stride = cfg.conv_stride, padding = cfg.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(cfg.maxpool_filter_size,cfg.maxpool_stride),
            
            nn.Conv2d(cfg.number_of_filters[2], cfg.number_of_filters[3], kernel_size = cfg.filter_size[3], stride = cfg.conv_stride, padding = cfg.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(cfg.maxpool_filter_size,cfg.maxpool_stride),
            
            #nn.Conv2d(cfg.number_of_filters[3],cfg.number_of_filters[4], kernel_size = cfg.filter_size[4], stride = cfg.conv_stride, padding = cfg.conv_padding),
            #nn.ReLU(),
            #nn.MaxPool2d(cfg.maxpool_filter_size,cfg.maxpool_stride),
            
            nn.Flatten(),
            
            nn.Linear(cfg.parameters_after_flatten,cfg.Linear_layers[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.Linear_layers[0], cfg.Linear_layers[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.Linear_layers[1],cfg.Linear_layers[2]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.Linear_layers[2],cfg.number_of_outputs)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    def fit(self, epochs, learning_rate, weight_decay, train_loader, val_loader, opt_func = torch.optim.AdamW):
    
        history = []
        optimizer = opt_func(self.parameters(),lr=learning_rate, weight_decay = weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            self.train()
            train_losses = []
            for batch in train_loader:
                loss = training_step(self, batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            result = evaluate(self, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            epoch_end(epoch, result)
            history.append(result)
    
        return history
    
    def test(self, test_loader):
        test_result = evaluate(self, test_loader) 
        
        return test_result
