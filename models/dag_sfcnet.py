"""
from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith bioRxiv 2019.12.17.879346
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCNet(nn.Module): # SFCN: Simple Fully Convolutional Network
    def __init__(
        self, channel_number, output_dim=2, dropout: float = 0.5,
    ):
        super(SFCNet, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            
            if n_layer == 2:
                if i < n_layer-1:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, 
                        maxpool_stride=12, max_kernel_size=(12,12,12),
                    ))
                else:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0,
                    ))
                avg_shape = [9, 11, 9]

            elif n_layer == 3:
                if i < n_layer-1:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, 
                        maxpool_stride=5, max_kernel_size=(5,5,5),
                    ))
                else:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0,
                    ))
                avg_shape = [4, 5, 4]
            elif n_layer == 4:
                if i < n_layer-1:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, 
                        maxpool_stride=3, max_kernel_size=(3,3,3),
                    ))
                else:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0,
                    ))
                avg_shape = [4, 5, 4]
            elif n_layer == 5:
                if i < n_layer-3:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, 
                        maxpool_stride=3, max_kernel_size=(3,3,3),
                    ))
                elif i < n_layer-1:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, 
                        maxpool_stride=2, max_kernel_size=(2,2,2),
                    ))
                else:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0,
                    ))
                avg_shape = [3, 3, 3]
            elif n_layer == 6:
                if i < n_layer-1:
                    self.feature_extractor.add_module('conv_%d' % i, self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1,
                        max_kernel_size=(2,2,2)))
                else:
                    self.feature_extractor.add_module('conv_%d' % i,
                                                      self.conv_layer(in_channel,
                                                                      out_channel,
                                                                      maxpool=False,
                                                                      kernel_size=1,
                                                                      padding=0))
                avg_shape = [3, 4, 3]
                self.cnn = nn.Sequential(
                    nn.Conv3d(1, channel_number[-2], kernel_size=3, stride=3, padding=1),
                    nn.BatchNorm3d(channel_number[-2]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=10, stride=10),
                )
                
        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        self.classifier.add_module('dropout', nn.Dropout(dropout))
        i = n_layer
        in_channel = channel_number[-1]
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, output_dim, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2, max_kernel_size=(2,2,2)):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(max_kernel_size, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x1 = self.feature_extractor(x)
        x2 = self.cnn(x)
        x = torch.add(x1, x1)
#         print(x1.size(), x2.size(), x.size())
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        
        x = x.view(x.size()[:2])
        
        return x

