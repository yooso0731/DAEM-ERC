# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:52:20 2023

@author: user
"""
import torch
import torch.nn as nn
from torchsummary import summary

## Attention (병렬)
class SENet(nn.Module):
  def __init__(self, in_channels, r=4):
    super().__init__()
    self.bn = nn.BatchNorm2d(in_channels)
    self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
    self.excitation = nn.Sequential(
        nn.Linear(in_channels, in_channels//r),
        nn.ReLU(),
        nn.Linear(in_channels//r, in_channels),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    shortcut = self.bn(x)
    x = self.squeeze(x)
    x = x.view(x.size(0), -1)
    x = self.excitation(x)
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = shortcut * x
    return x

class wavNet(nn.Module):
  def __init__(self, in_channels, num_classes, num_features):
    super().__init__()
    self.inputs = nn.Conv2d(1, in_channels, (1, 1), stride=(1, 1))

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels//4, (1, 1), stride=(1, 1)),
        nn.BatchNorm2d(in_channels//4),
        nn.Sigmoid(),
        nn.Conv2d(in_channels//4, in_channels, (1, 1), stride=(1, 1)),
        nn.BatchNorm2d(in_channels),
        nn.Sigmoid(),
        nn.Dropout(0.2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels//4, (3, 3), stride=(1, 1), padding='same'),
        nn.BatchNorm2d(in_channels//4),
        nn.Sigmoid(),
        nn.Conv2d(in_channels//4, in_channels, (3, 3), stride=(1, 1), padding='same'),
        nn.BatchNorm2d(in_channels),
        nn.Sigmoid(),
        nn.Dropout(0.2)
    )
    self.seblock = SENet(in_channels)
    
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels*2, (3, 3), stride=(1, 1), padding='valid'),
        nn.BatchNorm2d(in_channels*2),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.2)
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels*2, in_channels*4, (3, 3), stride=(1, 1), padding='valid'),
        nn.BatchNorm2d(in_channels*4),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.2)
    )
    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.do = nn.Dropout(0.2)
    self.features = nn.Linear(256, num_features)
    self.outputs = nn.Linear(num_features, num_classes)


  def forward(self, x):
    x = self.inputs(x)
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    x3 = self.seblock(x)

    x = x1 + x2 + x3

    x = self.relu(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.gap(x)
    features = x.view(x.size(0), -1)
    features = self.features(features)
    outputs = self.relu(features)
    outputs = self.do(outputs)
    outputs = self.outputs(outputs)

    return outputs