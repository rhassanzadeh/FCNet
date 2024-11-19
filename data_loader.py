import numpy as np
import pandas as pd
import nibabel as nib
# from scipy.io import loadmat

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


def get_data_loader(
    data: pd.core.frame.DataFrame, batch_size: int, num_workers: int, pin_memory: bool, drop_last:bool, 
    shuffle: bool, sampler=None, transform=None,
):
    dataset = MyDataset(data=data, transform=transform)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last, sampler = sampler)
    num_data = len(dataset)

    return (data_loader, num_data)


class MyDataset(Dataset): 
    def __init__(self, data, transform=None):
        self.data = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        subj_info = self.data.loc[idx]
        subj_dir = subj_info.smriPath
        
        label = subj_info.diagnosis
        
        # load image
        
#         data_info = loadmat(subj_dir)
            
#         if label == 1:
#             image = data_info['pre']["vbm_gm"].item()["dat"].item().astype('float32')
#         else:
#             image = data_info['session']["vbm_gm"].item()["dat"].item().astype('float32')
# #         image = image[:,:,28:86] # 58 slices

        image = nib.load(subj_dir.replace('.mat', '.nii.gz'))
#         image = nib.load(subj_dir.replace('.mat', '.nii'))
        image = image.get_fdata().astype('float32')
#         image = image[:,:,28:86] # 58 slices

        image = np.expand_dims(image, axis=0)        

        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        
        return (image, label, subj_info.ID)

    