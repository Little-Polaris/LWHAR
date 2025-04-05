import random

import numpy as np


def split_data(data: list[dict[str: str, str: int, str: int, str: int, str: np.ndarray[np.float32]]], config: dict, mode: str):
    main_performers = [i['main_performer'] for i in data]
    main_performers = np.array(main_performers, dtype=np.int8)
    performers = list(set(main_performers))
    if mode == 'pro1':
        train_performer_index = config['protocol1_train_performer_id']
        test_performer_index = list(set(performers) - set(train_performer_index))
    elif mode == 'pro2':
        train_performer_index = config['protocol2_train_performer_id']
        test_performer_index = list(set(performers) - set(train_performer_index))
    train_data_index = []
    test_data_index = []
    for i in train_performer_index:
        train_data_index.extend(list(np.where(main_performers == i)[0]))
    for i in test_performer_index:
        test_data_index.extend(list(np.where(main_performers == i)[0]))
    train_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['num_point'], config['dims'])).transpose(3, 0, 2, 1) for i in train_data_index]
    train_labels = [data[i]['label'] for i in train_data_index]
    train_filenames = [data[i]['filename'] for i in train_data_index]
    test_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['num_point'], config['dims'])).transpose(3, 0, 2, 1) for i in test_data_index]
    test_labels = [data[i]['label'] for i in test_data_index]
    test_filenames = [data[i]['filename'] for i in test_data_index]
    return {'data': train_data, 'label': train_labels, 'filename':train_filenames}, {'data': test_data, 'label': test_labels, 'filename':test_filenames}

