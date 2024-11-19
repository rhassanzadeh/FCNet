import numpy as np
import random
from os.path import join, exists

from utils import prepare_dirs, save_config
from config import get_config
from trainer import Trainer


def main(config):
    # set a standard random seed for reproducible results
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # ensure directories are setup
    trial_dir = join(config.logs_folder, config.task, config.model, 'trial'+str(config.trial_num))
    prepare_dirs(trial_dir, config.flush)
        
    if config.is_train and not config.resume:
        try:
            save_config(trial_dir, config)
        except ValueError:
            print(
                "[!] file already exist. Either change the trial number,",
                "or delete the json file and rerun.",
                sep=' ',
            )
    
    trainer = Trainer(trial_dir, config)

    if config.is_train: 
        trainer.train()
    else:
        trainer.test()

 
            
if __name__ == '__main__':    
    config, unparsed = get_config()
    main(config)
    