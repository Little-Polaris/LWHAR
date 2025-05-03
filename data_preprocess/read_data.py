import os
from typing import Any

import numpy as np

from UAV.UAVDataReader import UAVDataReader
from ntu60.NTU60DataReader import NTU60DataReader
from ntu120.NTU120DataReader import NTU120DataReader

def read_data(config: dict, filenames: list[str]=None) -> list[dict[str:int, str:int, str:int, str:int, str:np.ndarray[Any, np.float32]]]:
    if config['dataset_name'] == 'ntu60':
        data_reader = NTU60DataReader(config)
    elif config['dataset_name'] == 'ntu120':
        data_reader = NTU120DataReader(config)
    elif config['dataset_name'] == 'UAV':
        data_reader = UAVDataReader(config)
    if filenames is None:
        filenames = os.listdir(config['raw_dataset_dir'])
        if config['dataset_name'] == 'ntu120':
            filenames.extend(os.listdir(config['ntu60_raw_dataset_dir']))
    processed_data = data_reader.read_data(filenames)
    return processed_data