import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class AlexNet(nn.Module):
    def __init__(self, channel_number, num_classes: int = 2, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, channel_number[0], kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(channel_number[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(channel_number[0], channel_number[1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(channel_number[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(channel_number[1], channel_number[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(channel_number[2]),
            nn.ReLU(inplace=True),

            nn.Conv3d(channel_number[2], channel_number[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(channel_number[3]),
            nn.ReLU(inplace=True),

            nn.Conv3d(channel_number[3], channel_number[4], kernel_size=3, padding=1),
            nn.BatchNorm3d(channel_number[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
        # set up classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*channel_number[4], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_features=num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    