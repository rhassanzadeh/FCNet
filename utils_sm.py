from os.path import join, exists
import numpy as np
import os

from utils import load_config
from models import fcnet, alexnet, cnn, resnet

import torch



def load_checkpoint(trial_dir, model_type, repeat_num, split_num, device, best=False):
    params = load_config(trial_dir, repeat_num, split_num)
#     feature_size = params['feature_size']
    print('---------------------------')
    print('---------------------------')
    print('---------------------------')
    print(params)
    channel_number = params['channel_number']
    dropout = params['dropout']
    print(channel_number)
#     if model_type == 'AlexNet':
#         model = alexnet.AlexNet(num_classes=2, feature_size=feature_size)
#     elif model_type == 'ResNet':
#         model = resnet.resnet18(num_classes=2, feature_size=feature_size)
#     elif model_type == 'SFCNet':
#         model = fcnet.SFCNet(output_dim=2, channel_number=[32, 64, 128, 256, 256, feature_size])
    if model_type == 'SFCNet':
        model = fcnet.SFCNet(output_dim=2, channel_number=channel_number, dropout=dropout)
    elif model_type == 'har_SFCNet':
        model = fcnet.SFCNet(output_dim=2, channel_number=channel_number, dropout=dropout)
    # elif model_type == '113_SFCNetShallow':
    #     model = fcnet.SFCNet(
    #         output_dim=2, 
    #         channel_number=[32, 64, 128, 256, 256, feature_size], 
    #     )
    
    filename = f'model_ckpt_r{repeat_num}s{split_num}.tar'
    if best:
        filename = f'best_model_ckpt_r{repeat_num}s{split_num}.tar'
    print("[*] Loading model from {}".format(join(trial_dir,filename)))

    ckpt_path = join(trial_dir, filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    print(ckpt['model_state'].keys())
    
    model_dict = {k.replace("model.",""): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(model_dict)
    
    print(model_dict.keys())
        
    return model 


def min_max_normalization(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize between 0-1
    return image


from torch.autograd import Variable
import torch.nn.functional as F
def sensitivity_analysis(model, im, target_class=None, cuda=False, verbose=False, taskmode='clx'):
    # im nSubs x 1 x 121 x
    im = torch.Tensor(im)
    if cuda:
        im = im.cuda()
    X = Variable(im[None][None], requires_grad=True)
    output = model(X)
    if taskmode == 'clx':
        output = F.softmax(output, dim=1)
    # Backward pass.
    model.zero_grad()
    output_class = output.max(1)[1].data.cpu().numpy()[0]
    output_class_prob = output.max(1)[0].data[0].cpu().numpy()
    if verbose:
        print('Image was classified as', output_class,
              'with probability', output_class_prob, 'softmax output:', output)
    # one hot
    one_hot_output = torch.zeros(output.size())
    if target_class is None:
        one_hot_output[0, output_class] = 1
    else:
        one_hot_output[0, target_class] = 1
    if cuda:
        one_hot_output = one_hot_output.cuda()
        
    print('one_hot_output', one_hot_output)
    # Backward pass
    output.backward(gradient=one_hot_output)
    relevance_map = X.grad.data[0].cpu().numpy()
    return relevance_map


import nibabel as nib
def save_images(image: np.ndarray, file_path: str, file_name: str):
    """
        Saves image (cam activation map | gradient)
    Args:
        image: Numpy array of the image with shape (121, 145, 121)
        file_path: File path
        file_name: File name of the exported image
    """
    if not exists(file_path):
        os.makedirs(file_path)

    # Save image
    path_to_file = join(file_path, file_name)
#     np.save(path_to_file, image)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), join(file_path, f'{file_name}.nii'))

    
def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ()
        neg_saliency ()
    """
    pos_saliency = np.maximum(0, gradient)
    neg_saliency = np.maximum(0, -gradient)
#     pos_saliency = (np.maximum(0, gradient) / gradient.max())
#     neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency
    