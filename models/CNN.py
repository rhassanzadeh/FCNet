import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(
        self, channel_number, output_dim=2, dropout: float = 0.5,
    ):
        super(CNN, self).__init__()
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
        x = torch.randn(1, 113, 137, 113).view(-1, 1, 113, 137, 113)
        self._to_linear = None
        out = self.convs(x)
        print("output size of feature_extractor: ", self._to_linear)
        
        # define classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._to_linear, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_features=output_dim),
        ) 

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
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    def convs(self, x):      
        out = self.feature_extractor(x)
        if self._to_linear is None:
            print("input shape to fc1", out.shape)
            self._to_linear = out[0].size()[0]*out[0].size()[1]*out[0].size()[2]*out[0].size()[3]
        
        return out 
    

