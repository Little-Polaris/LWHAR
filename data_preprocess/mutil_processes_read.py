import os
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np

from read_data import read_data


def split_list(filenames, threads):
    n = len(filenames)
    size = n // threads
    remainder = n % threads

    result = []
    start = 0
    for i in range(threads):
        end = start + size + (1 if i < remainder else 0)
        result.append(filenames[start:end])
        start = end

    return result

def mutil_processes_read(config: dict) -> list[dict[str:int, str:int, str:int, str:int, str:np.ndarray[Any, np.float32]]]:
    dataset_dir = config['raw_dataset_dir']
    filenames = os.listdir(dataset_dir)
    process_num = config['process_num']

    data = []

    filenames = split_list(filenames, process_num)
    reader = partial(read_data, config)
    with Pool(processes=process_num) as pool:
        results = pool.map(reader, filenames)
    for i in range(process_num):
        data.extend(results[i])
    return data