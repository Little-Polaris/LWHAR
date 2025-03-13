import random

import numpy as np


def split_data(data: list[dict[str: str, str: int, str: int, str: int, str: np.ndarray[np.float32]]], config: dict, mode: str):
    train_data = []
    test_data = []
    if mode == 'CS':
        main_performers = [i['main_performer'] for i in data]
        main_performers = np.array(main_performers, dtype=np.int8)
        performers = list(set(main_performers))
        train_performer_index = performers[:int(len(performers) * 0.8)]
        test_performer_index = performers[int(len(performers) * 0.8):]
        train_data_index = []
        test_data_index = []
        for i in train_performer_index:
            train_data_index.extend(list(np.where(main_performers == i)[0]))
        for i in test_performer_index:
            test_data_index.extend(list(np.where(main_performers == i)[0]))
        train_data = [data[i] for i in train_data_index]
        test_data = [data[i] for i in test_data_index]
    elif mode == 'RAND':
        random.shuffle(data)
        train_data = data[:int(len(data) * 0.8)]
        test_data = data[int(len(data) * 0.8):]
    return train_data, test_data
