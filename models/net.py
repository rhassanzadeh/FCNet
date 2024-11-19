from models import FCNet, CNN, alexnet, resnet, dag, dag_sfcnet

import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
    
class Net(nn.Module):
    """backbone + projection head"""
    def __init__(self, model_type, channel_number=[], num_classes: int = 2, feature_size: int = 64, dropout: float = 0.5):
        super(Net, self).__init__()

        if model_type == 'FCNet':
            self.model = FCNet.FCNet(
                output_dim=num_classes, 
                channel_number=channel_number, 
                dropout=dropout,
            )
        elif model_type == 'har_FCNet':
            self.model = FCNet.FCNet(
                output_dim=num_classes, 
                channel_number=channel_number, 
                dropout=dropout,
            )
        elif model_type == 'CNN':
            self.model = CNN.CNN(
                output_dim=num_classes, 
                channel_number=channel_number, 
                dropout=dropout,
            )
        elif model_type == 'dag_sfcnet':
            self.model = dag_sfcnet.SFCNet(
                output_dim=num_classes, 
                channel_number=channel_number, 
                dropout=dropout,
            )
        elif model_type == 'dag':
            self.model = dag.DAG(
                channel_number=channel_number,
                num_classes=num_classes, 
                dropout=dropout,
            )
        elif model_type == 'AlexNet':
            self.model = alexnet.AlexNet(
                channel_number=channel_number,
                num_classes=num_classes, 
                dropout=dropout,
            )
        elif model_type == 'ResNet':
            self.model = resnet.resnet18(
                num_classes=num_classes, feature_size=feature_size, dropout=dropout,
            )
        else:
            raise Exception("Wrong model.")

    def forward(self, x):
        out = self.model(x)
        return out
