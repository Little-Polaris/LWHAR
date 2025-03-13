import os
import random

import numpy as np
import torch

from Feeder.ntu60_feeder import Feeder


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def MyDataLoader(config, device):
    train_data_loader = torch.utils.data.DataLoader(
                dataset=Feeder(data_path=os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_cs_processed_data.pkl'),
                               label_path=None,
                               p_interval=config['train_p_interval'],
                               split='train',
                               random_choose=False,
                               random_shift=False,
                               random_move=False,
                               random_rot=True,
                               window_size=64,
                               normalization=False,
                               debug=False,
                               use_mmap=False,
                               bone=False,
                               vel=False),
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['dataloader_num_workers'],
                drop_last=True,
                worker_init_fn=init_seed)

    test_data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_cs_processed_data.pkl'),
                           label_path=None,
                           p_interval=config['test_p_interval'],
                           split='test',
                           random_choose=False,
                           random_shift=False,
                           random_move=False,
                           random_rot=False,
                           window_size=64,
                           normalization=False,
                           debug=False,
                           use_mmap=False,
                           bone=False,
                           vel=False),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['dataloader_num_workers'],
            drop_last=False,
            worker_init_fn=init_seed)
    return train_data_loader, test_data_loader