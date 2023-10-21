# -*- coding: utf-8 -*-
"""
Functions for training a neural network model

@author: Sazzad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


def training_step(model, batch):
    images, labels = batch
    out = model(images)     # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    print('Predictions: ', preds)
    print('Original Labels: ', labels)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
def validation_step(model, batch):
    images, labels = batch 
    out = model(images)                    # Generate predictions
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}
        
def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def epoch_end(epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        outputs = [validation_step(model, batch) for batch in val_loader]
        return validation_epoch_end(outputs)
    
 