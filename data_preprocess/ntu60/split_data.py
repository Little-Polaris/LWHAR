import numpy as np


def split_data(data: dict[str: str, str: int, str: int, str: int, str: np.ndarray[np.float32]], config: dict, mode: str):

    if mode == 'CS':
        train_performer_index = config['train_performer_id']
        test_performer_index = config['test_performer_id']

        main_performers = [i['main_performer'] for i in data]
        main_performers = np.array(main_performers, dtype=np.int8)
        train_data_index = []
        test_data_index = []
        for i in train_performer_index:
            train_data_index.extend(list(np.where(main_performers == i)[0]))
        for i in test_performer_index:
            test_data_index.extend(list(np.where(main_performers == i)[0]))
    elif mode == 'CV':
        train_camera_index = config['train_camera_id']
        test_camera_index = config['test_camera_id']

        cameras = [i['camera'] for i in data]
        cameras = np.array(cameras, dtype=np.int8)

        train_data_index = []
        test_data_index = []

        for i in train_camera_index:
            train_data_index.extend(list(np.where(cameras == i)[0]))
        for i in test_camera_index:
            test_data_index.extend(list(np.where(cameras == i)[0]))

    train_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['node_num'], config['dims'])).transpose(3, 0, 2, 1) for i in train_data_index]
    train_labels = [data[i]['label'] for i in train_data_index]
    test_data = [data[i]['data'].reshape((data[i]['data'].shape[0], 2, config['node_num'], config['dims'])).transpose(3, 0, 2, 1) for i in test_data_index]
    test_labels = [data[i]['label'] for i in test_data_index]
    return {'data': train_data, 'label': train_labels}, {'data': test_data, 'label': test_labels}