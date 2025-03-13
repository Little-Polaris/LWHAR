import argparse
import json
import os
import pickle
from datetime import datetime

from data_preprocess.UAV.split_data import split_data as uav_split_data
from data_preprocess.ntu60.split_data import split_data as ntu60_split_data
from mutil_processes_read import mutil_processes_read
from read_data import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help='path of dataset preprocess configuration file ')
    args = vars(parser.parse_args())
    with open(args['config_path'], 'r') as file:
        config = dict(json.load(file))

    processed_data = []

    start_time = datetime.now()
    if 'process_num' in config and config['process_num'] > 1:
        processed_data = mutil_processes_read(config)
    else :
        processed_data = read_data(config)
    processed_data = sorted(processed_data, key=lambda x: x['filename'])

    if not os.path.exists(config['binary_data_dir']) :
        os.makedirs(config['binary_data_dir'], 0o777)

    if config['dataset_name'] == 'ntu60':
        cs_train_data, cs_test_data = ntu60_split_data(processed_data, config, 'CS')
        cv_train_data, cv_test_data = ntu60_split_data(processed_data, config, 'CV')
        with open(os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_cs_processed_data.pkl'), 'wb') as f:
            pickle.dump({'cs_train_data': cs_train_data, 'cs_test_data': cs_test_data}, f)
        with open(os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_cv_processed_data.pkl'), 'wb') as f:
            pickle.dump({'cv_train_data': cv_train_data, 'cv_test_data': cv_test_data}, f)
    elif config['dataset_name'] == 'UAV':
        cs_train_data, cs_test_data = uav_split_data(processed_data, config, 'CS')
        rand_train_data, rand_test_data = uav_split_data(processed_data, config, 'RAND')
        with open(os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_cs_processed_data.pkl'), 'wb') as f:
            pickle.dump({'cs_train_data': cs_train_data, 'cs_test_data': cs_test_data}, f)
        with open(os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_rand_processed_data.pkl'), 'wb') as f:
            pickle.dump({'rand_train_data': rand_train_data, 'rand_test_data': rand_test_data}, f)

    end_time = datetime.now()
    duration = end_time - start_time
    print(duration)
