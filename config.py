import argparse


def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='argument for training')

parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs')

# path and dirs
parser.add_argument('--logs_folder', type=str, default='./logs/',
                    help='folder in where logs and checkpoints are stored')

# optimization
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=4e-3,
                    help='weight decay')
parser.add_argument('--optimizer', type=str, default='SGD', 
                    help='Adam | SGD')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--early_stop', type=str2bool, default=True,
                    help='early stops the training if validation loss/accuracy does not improve after a given patience.')
parser.add_argument('--flush', type=str2bool, default=False,
                    help='whether to delete ckpt + log files for trial')
parser.add_argument('--best', type=str2bool, default=True,
                    help='load best model or most recent for testing')
parser.add_argument('--random_seed', type=int, default=42,
                    help='general seed to ensure reproducibility')
parser.add_argument('--split_num', type=int, default=42,
                    help='split number of data')
parser.add_argument('--repeat_num', type=int, default=42,
                    help='repeat number of data')
parser.add_argument('--resume', type=str2bool, default=False,
                    help='whether to resume training from checkpoint')
parser.add_argument('--is_train', type=str2bool, default=True,
                    help='whether to train or test the model')
parser.add_argument('--train_patience', type=int, default=20,
                    help='number of epochs to wait before stopping training')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='specifies the amount of smoothing when computing the cross entropy loss')
parser.add_argument('--augmentation', type=str2bool, default=False,
                    help='whether to augment')
parser.add_argument('--harmonize', type=str2bool, default=False,
                    help='whether to harmonize data')
parser.add_argument('--model', type=str, default='CNN',
                    help='CNN | ResNet | AlexNet')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout')
parser.add_argument('--feature_size', type=int, default=64,
                    help='feature size')
parser.add_argument('--task', type=str, default='lateralization',
                    help='classification or regression task: lateralization | Gender | EP_CN | ...')
parser.add_argument('--trial_num', type=int, default=1, required=True,
                    help='trial number when recording multiple runs')
parser.add_argument('--channel_number', type=lambda s: [int(item) for item in s.split(',')],
                    help='Channels size') # input in cmp: --channel_number "128,64" or 128,64


def get_config():
    config, unparsed = parser.parse_known_args()
    
    return config, unparsed

