"""
from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith bioRxiv 2019.12.17.879346
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAG(nn.Module): 
    def __init__(
        self, channel_number, num_classes=2, dropout: float = 0.5,
    ):
        super(DAG, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, channel_number[0], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(channel_number[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(channel_number[0], channel_number[1], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(channel_number[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(channel_number[1], channel_number[2], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(channel_number[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv3d(1, channel_number[2], kernel_size=8, stride=8, padding=0),
            nn.BatchNorm3d(channel_number[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=8, stride=8),
        )
        # set up classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*channel_number[2], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_features=num_classes),
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x = torch.add(x1, x1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

