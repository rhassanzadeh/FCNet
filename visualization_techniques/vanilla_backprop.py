'''
from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master
'''

import sys, pickle, pathlib, json
import pandas as pd
from os.path import join
import numpy as np
import nibabel as nib
from glob import glob
from statistics import mean

sys.path.insert(1, '/data/users1/reihaneh/projects/emory-epilepsy-aphasia/3D')
from utils import load_checkpoint_sm, save_images

import torch
import torch.nn.functional as F


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
#         first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.model.feature_extractor.conv_0._modules.items())[0][1]
        first_layer.register_full_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class, verbose=False):
        # Forward
        output = self.model(input_image.requires_grad_(True))
        # Zero grads
        self.model.zero_grad()
        if verbose:
            output_class = output.max(1)[1].data.cpu().numpy()[0]
            output_class_prob = output.max(1)[0].data[0].cpu().numpy()
            print('Image was classified as', output_class,
                  'with probability', output_class_prob, 
                  'softmax output:', output,
                  'True label', target_class,
                 )
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(self.device)
        one_hot_output[0][target_class] = 1
        print('one_hot_output: ', one_hot_output)
        # Backward pass
        output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the dimension, i.e., channel (1, 113, 137, 113)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr


    
if __name__ == '__main__':
    vis_technique = sys.argv[0].split('/')[-1].split('.')[0]
    subj_id = sys.argv[1]
    image_dir = sys.argv[2]
    label = int(sys.argv[3])
    task = sys.argv[4]
    model_type = sys.argv[5]
    repeat_num = int(sys.argv[6])
    split_num = int(sys.argv[7])
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load image
    img = np.float32(nib.load(image_dir).get_fdata()) # shape = (113, 137, 113)
    img = np.expand_dims(img, axis=(0, 1)) # reshape to (batch_size, channel, width, height, depth): (1, 1, 113, 137, 113)
    img = torch.from_numpy(img).to(device) # set the device for the image

    logs_dir = f'logs/{task}/{model_type}'
    all_files = glob(join(logs_dir,'trial[0-9]*'))
    best_valid_acc = 0
    best_trial = None
    
    for ind, fle in enumerate(all_files):
        trial_num = fle.split('/')[-1]
        valid_accs = []
        metric_logs = json.load(open(join(join(logs_dir, trial_num), f'logs_r{repeat_num}s{split_num}.json')))
        valid_accs.append(metric_logs[f'best_valid_acc'])
        if best_valid_acc < mean(valid_accs):
            best_valid_acc = mean(valid_accs)
            best_trial = trial_num

    all_grads = []
    trial_dir = join(logs_dir, best_trial)
    # load model
    pretrained_model = load_checkpoint_sm(trial_dir=trial_dir, model_type=model_type, 
                                       repeat_num=repeat_num, split_num=split_num, device=device, best=True).to(device)
    # vanilla backprop
    VBP = VanillaBackprop(pretrained_model, device)
    # generate gradients
    vanilla_grads = VBP.generate_gradients(img, label, verbose=True)

    all_grads.append(vanilla_grads)
        
    mean_grad = np.mean(all_grads, axis=0) # shape = (1, 113, 137, 113)
    mean_grad = np.squeeze(mean_grad) # shape = (113, 137, 113)
    print('mean_grad.shape', mean_grad.shape)
    
    fig_path = join('/data/users1/reihaneh/projects/emory-epilepsy-aphasia/3D/save/saliency_maps', task, model_type, vis_technique, f'r{repeat_num}')
    pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True) 
    # Save mask
    save_images(mean_grad, fig_path, subj_id)
    print('Vanilla backprop completed')
    