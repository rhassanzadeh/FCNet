import numpy as np
import os, time, shutil, json, random
from tqdm import tqdm
from os.path import join, exists
import pandas as pd
from statistics import mean, stdev
import torchio as tio
from pathlib import Path
from collections import Counter
from glob import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from utils import *
from config import get_config
from models.net import Net
from data_loader import get_data_loader

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics.classification import BinaryAUROC
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, trial_dir, config):
        self.trial_dir = trial_dir
        self.config = config
        
        if self.config.is_train:
            print(self.config)

        # set device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_model(self, class_weights:torch.Tensor=None, label_smoothing:float=0.0):
        # set model and loss function
        model = Net(
            model_type=self.config.model, channel_number=self.config.channel_number,
            feature_size=self.config.feature_size, dropout=self.config.dropout,
        )           
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
                
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print('# of gpus available:' , torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.to(self.device)
        criterion = criterion.to(self.device)
        cudnn.benchmark = (self.device=='cuda')
        
        # model params
        if self.config.is_train:
            num_params = sum([p.data.nelement() for p in model.parameters()])
            num_layers = len(list(model.children()))
            print()
            print("-----------------model-----------------")
            print(model)
            print("---------------------------------------")
            print('[*] Number of model parameters: {:,}'.format(num_params))
            print()

        return model, criterion
    
    def set_optimizer(self):
        if self.config.optimizer == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, 
                                  momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

#         scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        return optimizer, scheduler

    def train(self):  
        
        X_train = pd.read_pickle(
            f'/data/users1/reihaneh/projects/emory-epilepsy-aphasia/save/new_splits2/train_r{self.config.repeat_num}s{self.config.split_num}.pkl'
        )
        if self.config.harmonize:
            X_train['smriPath'] = X_train['smriPath'].str.replace(
                '/data/users1/reihaneh/data/Epilepsy/Completed_Request/Controls/processed/',
                f'/data/users1/reihaneh/projects/emory-epilepsy-aphasia/save/harmonization/r{self.config.repeat_num}s{self.config.split_num}/data/'
            )
            X_train['smriPath'] = X_train['smriPath'].str.replace(
                '/data/users1/reihaneh/data/Epilepsy/Completed_Request/Patients/processed/',
                f'/data/users1/reihaneh/projects/emory-epilepsy-aphasia/save/harmonization/r{self.config.repeat_num}s{self.config.split_num}/data/'
            )
        X_train, X_valid, _, _ = train_test_split(X_train, X_train.diagnosis, test_size=0.15, 
                                                  stratify=X_train.diagnosis, random_state=self.config.random_seed)

        # class weights for passing to the loss
        num_class1 = len(X_train.loc[X_train['diagnosis']==0])
        num_class2 = len(X_train.loc[X_train['diagnosis']==1])
        class_weights = torch.tensor([1-num_class1/X_train.shape[0], 1-num_class2/X_train.shape[0]])

        # compute samples weight for stratifying data in dataloader and oversamplying
        count=Counter(X_train.diagnosis)
        class_count=np.array([count[0],count[1]])
        weight=1./class_count
        samples_weight = np.array([weight[t] for t in X_train.diagnosis])
        samples_weight=torch.from_numpy(samples_weight)
        # sampling based on the passed weights
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), 
            replacement=False, # in case of having imbalanced dataset, set replacement to True for oversampling
        )

        perf_logs = {} # model performance logs
        logs_path = join(self.trial_dir, f'logs_r{self.config.repeat_num}s{self.config.split_num}.json')

        # composition of transforms used to train the model
        if self.config.augmentation:
#             training_transform=tio.OneOf([
#                     tio.RandomFlip(axes=('LR',)), # flip along lateral axis only
#                     tio.RandomAffine(scales=(0.9, 1.2), degrees=15),
#                     tio.RandomMotion(num_transforms=2, image_interpolation='nearest'), # image_interpolation='linear'
#                     tio.RandomGhosting(intensity=1.0),
#                     tio.RandomBlur(),
#                     tio.RandomNoise(std=0.5),
#                     tio.RandomBiasField(coefficients=1)
#                 ]),
            training_transform = tio.Compose([
#                 tio.RandomAnisotropy(p=0.1),               # make images look anisotropic 20% of times
#                 tio.CropOrPad((110, 135, 121)),          # tight crop around brain
                tio.RandomBlur(p=0.1),                     # blur 20% of times
                tio.RandomNoise(p=0.7),                    # Gaussian noise 20% of times
#                 tio.OneOf({                                # either
#                     tio.RandomAffine(degrees=6): 0.8,      # random affine
#                     tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
#                 }, p=0.1),                                 # applied to 50% of images
#                 tio.RandomBiasField(p=0.1),                # magnetic field inhomogeneity 30% of times
#                 tio.OneOf({                                # either
#                     tio.RandomMotion(): 2,                 # random motion artifact
#                     tio.RandomSpike(): 1,                  # or spikes
#                     tio.RandomGhosting(): 2,               # or ghosts
#                 }, p=0.1),                                 # applied to 50% of images
            ])
        else:
            training_transform = None

        # build data loaders
        train_loader, num_train = get_data_loader(
            data=X_train, batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
#             shuffle=False, sampler=sampler, # shuffle=True if sampler=None
            shuffle=True,
            transform=training_transform,
        )
        valid_loader, num_valid = get_data_loader(
            data=X_valid, batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            pin_memory=True, drop_last=False,
            shuffle=True, 
        )

        print("\n====================================================")
        print('\t repeat {} - split {}'.format(self.config.repeat_num, self.config.split_num))
        print("====================================================")
        print(f'totla number of subjects: {len(X_train)+len(X_valid)}')
        print(f'running on {num_train} train samples and {num_valid} validiation samples')

        self.start_epoch = 1 
        # early stopping params
        self.best_valid_acc = None
        self.counter = 0

        # build model and criterion
        self.model, self.criterion = self.setup_model(
#             class_weights=class_weights, 
            label_smoothing=self.config.label_smoothing,
        )
        # build optimizer
        self.optimizer, self.scheduler = self.set_optimizer()

        if self.config.resume:
            self.load_checkpoint(repeat_num=self.config.repeat_num, split_num=self.config.split_num, best=False)
            if exists(logs_path):
                with open(logs_path, 'r+') as f:
                    perf_logs = json.load(f)

        for epoch in range(self.start_epoch, self.config.epochs + 1):

            train_loss, train_accuracy, train_time = self.train_one_epoch(train_loader, num_train)
            valid_loss, valid_accuracy = self.validate(valid_loader)
            
#             self.scheduler.step()
            self.scheduler.step(valid_loss)

            # check for improvement
            if self.best_valid_acc is None:
                self.best_valid_acc = valid_accuracy
                is_best = True
            else:
                is_best = valid_accuracy > self.best_valid_acc
            msg = "Epoch:{}, {:.1f}s - train loss/accuracy: {:.4f}/{:.4f} - validation loss/accuracy: {:.4f}/{:.4f}"
            if is_best:
                msg += " [*]"
                self.counter = 0
            print(msg.format(epoch, train_time, train_loss, train_accuracy, valid_loss, valid_accuracy))

            # checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.config.train_patience and self.config.early_stop:
                print("[!] No improvement in a while, stopping training.")
                break
            self.best_valid_acc = max(valid_accuracy, self.best_valid_acc)

            self.save_checkpoint(
            {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'best_valid_acc': self.best_valid_acc,
                'counter': self.counter,
            }, repeat_num=self.config.repeat_num, split_num=self.config.split_num, is_best=is_best)

            # log performance
            perf_logs.update({
                f'best_valid_acc': self.best_valid_acc,
                f'epoch_{epoch}_train_loss': train_loss,
                f'epoch_{epoch}_train_accuracy': train_accuracy,
                f'epoch_{epoch}_valid_loss': valid_loss,
                f'epoch_{epoch}_valid_accuracy': valid_accuracy,
            })
            with open(logs_path, 'w') as fp:
                json.dump(perf_logs, fp, indent=4, sort_keys=False)

        print("\ndone!")

    def train_one_epoch(self, train_loader, num_train):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        batch_time.reset
        losses.reset
        accuracies.reset
        self.model.train()
        tic = time.time()
        
        # switch to train mode
        self.model.train()
        
        with tqdm(total=num_train) as pbar:            
            for batch_index, (inputs, targets) in enumerate(train_loader):  
                batch_size = inputs.shape[0]

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # compute loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update metric
                losses.update(loss.item(), batch_size)
                acc = (torch.argmax(outputs, 1) == targets).float().mean().item()*100
                accuracies.update(acc, batch_size) 

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(("{:.1f}s - loss: {:.3f}".format(batch_time.val, losses.val)))
                pbar.update(batch_size)

        return losses.avg, accuracies.avg, batch_time.sum

    def validate(self, valid_loader):
        losses = AverageMeter()
        accuracies = AverageMeter()
        losses.reset
        accuracies.reset
        
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(valid_loader):
                batch_size = inputs.shape[0]

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # compute loss and accuracy
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets).item()
                acc = (torch.argmax(outputs, 1) == targets).float().mean().item()*100
                
                # store batch statistics
                losses.update(loss, batch_size)
                accuracies.update(acc, batch_size) 

        return losses.avg, accuracies.avg


    def test(self): 
        import warnings
        warnings.filterwarnings('ignore')
                
        test_metric_logs = {}
        test_metric_dir = f'./save/test_logs'
        Path(test_metric_dir).mkdir(parents=True, exist_ok=True)
        test_metric_file = f'{test_metric_dir}/{self.config.task}_{self.config.model}_test_metrics.json'
        
        # find optimal model for each test set
        logs_dir = f'logs/{self.config.task}/{self.config.model}'
        all_files = glob(join(logs_dir,'trial[0-9]*'))  
        
        # find best model on validation set
        best_valid_acc = 0
        best_trial = None
        for ind, fle in enumerate(all_files):
            trial_num = fle.split('/')[-1]
            valid_accs = []
            config_params = load_config(fle, repeat_num=self.config.repeat_num, split_num=self.config.split_num)
            metric_logs = json.load(open(
                join(join(logs_dir, trial_num), f'logs_r{self.config.repeat_num}s{self.config.split_num}.json')
            ))
            
            if best_valid_acc < metric_logs[f'best_valid_acc']:
                best_valid_acc = metric_logs[f'best_valid_acc']
                best_trial = trial_num
#                 best_params = config_params
                 
        print("======== best config ============")
        params = load_config(join(logs_dir, best_trial), self.config.repeat_num, self.config.split_num)
        self.config = RecursiveNamespace(**params)
        self.config.is_train = False
        self.trial_dir = self.trial_dir.replace("trial1", f"trial{self.config.trial_num}")
        print(self.config)
        
        # load test data
        X_test = pd.read_pickle(
            f'/data/users1/reihaneh/projects/emory-epilepsy-aphasia/save/new_splits2/test_r{self.config.repeat_num}s{self.config.split_num}.pkl'
        )
        if self.config.harmonize:
            X_test['smriPath'] = X_test['smriPath'].str.replace(
                '/data/users1/reihaneh/data/Epilepsy/Completed_Request/Controls/processed/',
                f'/data/users1/reihaneh/projects/emory-epilepsy-aphasia/save/harmonization/r{self.config.repeat_num}s{self.config.split_num}/data/'
            )
        test_loader, num_test = get_data_loader(
            data=X_test, batch_size=4, num_workers=0, shuffle=False, pin_memory=True, drop_last=False,
        )  
 
        # load best model
        accuracies = AverageMeter()
        accuracies.reset 

        # build model
        self.model, _ = self.setup_model()
        # build optimizer
        self.optimizer, _ = self.set_optimizer() 
        # load best model
        self.load_checkpoint(repeat_num=self.config.repeat_num, split_num=self.config.split_num, best=True)
        # switch to evaluate mode
        self.model.eval()

        t_TP, t_TN, t_FP, t_FN = 0.0, 0.0, 0.0, 0.0
        num_classes = 2
        metric = BinaryAUROC(thresholds=None)
        pred_df = pd.DataFrame(columns=['subjectID', f'prediction (r{self.config.repeat_num}_s{self.config.split_num})'])
        y_score = []
        labels = []
        with torch.no_grad():
            for batch_index, (inputs, targets, IDs) in enumerate(test_loader):
                batch_size = inputs.shape[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # compute accuracy
                outputs = self.model(inputs)
                        
                probabilities = F.softmax(outputs, dim=1)[:, 1]
                y_score.extend(list(probabilities.cpu().numpy()))
                labels.extend(list(targets.cpu().numpy()))
                
#                 print(labels, y_score)
                
                new_data = {'subjectID': IDs, f'prediction (r{self.config.repeat_num}_s{self.config.split_num})': (torch.argmax(outputs, 1) == targets).float().cpu()}
                pred_df = pred_df.append(pd.DataFrame(new_data), ignore_index=True)

                acc = (torch.argmax(outputs, 1) == targets).float().mean().item()*100
                
                if batch_index == 0:
                    all_outputs = torch.argmax(outputs, 1)
                    all_targets = targets
                else:
                    all_outputs = torch.cat((all_outputs, torch.argmax(outputs, 1)), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                
                # create a confusion matrix
                conf_matrix = torch.zeros(num_classes, num_classes)
                for t, o in zip(targets, torch.argmax(outputs, 1)):
                    conf_matrix[t, o] += 1
                    
#                 print(conf_matrix)

                TP = conf_matrix.diag()
                for c in range(num_classes):
                    idx = torch.ones(num_classes).byte()
                    idx[c] = 0
                    # all non-class samples classified as non-class
                    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
                    # all non-class samples classified as class
                    FP = conf_matrix[idx, c].sum()
                    # all class samples not classified as class
                    FN = conf_matrix[c, idx].sum()

                    if c==1:
                        t_TP += TP[c]
                        t_TN += TN
                        t_FP += FP
                        t_FN += FN
                

                # store batch statistics
                accuracies.update(acc, batch_size) 

        sens = (t_TP / (t_TP + t_FN) * 100).item()
        spec = (t_TN / (t_TN + t_FP) * 100).item()
        ppv = (t_TP / (t_TP + t_FP) * 100).item()
        npv = (t_TN / (t_TN + t_FN) * 100).item()
        print("acc:", accuracies.avg)
        print("sens:", sens, ", spec:", spec)
        print("PPV:", ppv, ", NPV:", npv)
        auc_ = metric(all_outputs.cpu(), all_targets.cpu()).item() * 100
        print("auc:", auc_)
        print()

        nn_fpr, nn_tpr, nn_thresholds = roc_curve(labels, y_score)
        
        if exists("./save/test_logs/ROC.json"):
            with open("./save/test_logs/ROC.json", "r") as jsonFile:
                ROCdata = json.load(jsonFile)
        else:
            ROCdata = {}
        ROCdata.update({
            f"fpr_r{self.config.repeat_num}_s{self.config.split_num}": nn_fpr.tolist(),
            f"tpr_r{self.config.repeat_num}_s{self.config.split_num}": nn_tpr.tolist(),
            f"threshold_r{self.config.repeat_num}_s{self.config.split_num}": nn_thresholds.tolist(),
        })
        with open("./save/test_logs/ROC.json", "w") as fp:
            json.dump(ROCdata, fp, indent=4, sort_keys=False)
        
#         roc_auc = auc(nn_fpr, nn_tpr)
#         plt.title('Receiver Operating Characteristic')
#         plt.plot(nn_fpr, nn_tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#         plt.legend(loc = 'lower right')
#         plt.plot([0, 1], [0, 1],'r--')
#         plt.xlim([0, 1])
#         plt.ylim([0, 1])
#         plt.ylabel('True Positive Rate')
#         plt.xlabel('False Positive Rate')

#         plt.savefig("./ROC.png",  bbox_inches ="tight", pad_inches = 1, transparent = True)
                
        pred_df.to_csv(f'./save/test_logs/predictions/pred_r{self.config.repeat_num}_s{self.config.split_num}.csv', index=False)
                
        if exists(test_metric_file):
            with open(test_metric_file, 'r+') as f:
                test_metric_logs = json.load(f)

        test_metric_logs.update({
            f'accuracy_r{self.config.repeat_num}s{self.config.split_num}': accuracies.avg, 
            f'sensitivity_r{self.config.repeat_num}s{self.config.split_num}': spec, 
            f'specificity_r{self.config.repeat_num}s{self.config.split_num}': sens,
            f'PPV_r{self.config.repeat_num}s{self.config.split_num}': ppv,
            f'NPV_r{self.config.repeat_num}s{self.config.split_num}': npv,
            f'AUC_r{self.config.repeat_num}s{self.config.split_num}': auc_,
        })

        with open(test_metric_file, 'w') as fp:
            json.dump(test_metric_logs, fp, indent=4, sort_keys=False)
        
    def save_checkpoint(self, state, repeat_num, split_num, is_best):
        filename = f'model_r{repeat_num}s{split_num}.tar'
        ckpt_path = join(self.trial_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = f'best_model_ckpt_r{repeat_num}s{split_num}.tar'
            shutil.copyfile(
                ckpt_path, join(self.trial_dir, filename)
            )
            
    def load_checkpoint(self, repeat_num, split_num, best=False):
        model_dir = self.trial_dir
        print(f'------- repeat {repeat_num} - split {split_num} ----------------')
        print("[*] Loading model from {}".format(model_dir))

        filename = f'model_ckpt_r{repeat_num}s{split_num}.tar'
        if best:
            filename = f'best_model_ckpt_r{repeat_num}s{split_num}.tar'
        ckpt_path = join(model_dir, filename)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch'] + 1
        self.best_valid_acc = ckpt['best_valid_acc']
        self.counter = ckpt['counter']
        self.optimizer.load_state_dict(ckpt['optim_state'])
        
        if torch.cuda.device_count() > 1:
            self.model.load_state_dict(ckpt['model_state'])
        else:
            model_dict = {k.replace("module.",""): v for k, v in ckpt['model_state'].items()}
            self.model.load_state_dict(model_dict)

#         model_dict = {f"module.{k}": v for k, v in ckpt['model_state'].items()}
#         self.model.load_state_dict(model_dict)
        
        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} with best valid accuracy of {:.4f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {} with best valid accuracy of {:.4f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
            
              