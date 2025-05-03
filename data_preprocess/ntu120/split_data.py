import pickle

import numpy as np


def split_data(data: dict[str: str, str: int, str: int, str: int, str: np.ndarray[np.float32]], config: dict, mode: str):

    if mode == 'CSub':
        train_performer_index = config['train_performer_id']
        test_performer_index = [i for i in range(1, 107) if i not in train_performer_index]

        main_performers = [i['main_performer'] for i in data]
        main_performers = np.array(main_performers, dtype=np.int8)
        train_data_index = []
        test_data_index = []
        for i in train_performer_index:
            train_data_index.extend(list(np.where(main_performers == i)[0]))
        for i in test_performer_index:
            test_data_index.extend(list(np.where(main_performers == i)[0]))
        with open('../datasets/nturgb+d_skeletons/preprocessed/ntu60_cs_processed_data.pkl', 'rb') as f:
            ntu60_data = pickle.load(f)
            ntu60_data['train_data'] = ntu60_data['cs_train_data']
            ntu60_data['test_data'] = ntu60_data['cs_test_data']
            del ntu60_data['cs_train_data']
            del ntu60_data['cs_test_data']
    elif mode == 'CSet':
        train_set_index = config['train_set_id']
        test_set_index = config['test_set_id']

        sets = [int(i['filename'][1:4]) for i in data]
        sets = np.array(sets, dtype=np.int8)

        train_data_index = []
        test_data_index = []

        for i in train_set_index:
            train_data_index.extend(list(np.where(sets == i)[0]))
        for i in test_set_index:
            test_data_index.extend(list(np.where(sets == i)[0]))

    train_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['num_point'], config['dims'])).transpose(3, 0, 2, 1) for i in train_data_index]
    train_labels = [data[i]['label'] for i in train_data_index]
    test_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['num_point'], config['dims'])).transpose(3, 0, 2, 1) for i in test_data_index]
    test_labels = [data[i]['label'] for i in test_data_index]
    return {'data': train_data, 'label': train_labels}, {'data': test_data, 'label': test_labels}
