import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
from numpy import dtype, ndarray


class DataReader(ABC):
    def __init__(self, config: dict):
        self.dataset_name = config["dataset_name"]
        self.raw_dataset_dir = config['raw_dataset_dir']
        self.to_be_removed_files = config['to_be_removed_files']
        self.binary_data_dir = config['binary_data_dir']
        self.file_suffix = config['file_suffix']
        self.dims = config['dims']
        self.node_num = config['node_num']
        self.invalid_frame_num_lower_bound = config['invalid_frame_num_lower_bound']
        self.camera_start = config['camera_start']
        self.camera_end = config['camera_end']
        self.performer_start = config['performer_start']
        self.performer_end = config['performer_end']
        self.label_start = config['label_start']
        self.label_end = config['label_end']
        self.mutil_processes = config['process_num']
        self.aligned_frame_num = config['aligned_frame_num']
        self.train_p_interval = config['train_p_interval']
        self.test_p_interval = config['test_p_interval']
        self.NOISE_LEN_THRESHOLD = 11 if 'noise_length_threshold' not in config else config['noise_length_threshold']
        self.NOISE_SPR_THRESHOLD1 = 0.8 if 'noise_spr_threshold1' not in config else config['noise_spr_threshold1']
        self.NOISE_SPR_THRESHOLD2 =  1 - 0.69754 if 'noise_spr_threshold2' not in config else config['noise_spr_threshold2']

        self.raw_data: list[list[str]] = []
        self.processed_data: list[dict[str: int, str:int, str:int, str:int, str:np.ndarray[Any, np.float32]]] = []


    @abstractmethod
    class Joint:
        RIGHT_SHOULDER = None
        LEFT_SHOULDER = None
        RIGHT_HIP = None
        LEFT_HIP = None

    def read_data_from_disk(self, filenames: list[str]):
        for filename in filenames:
            with open(os.path.join(self.raw_dataset_dir, filename), 'r') as file:
                self.raw_data.append(file.readlines())

    @abstractmethod
    def read_file(self, file_content: list[str], dims: int):
        pass

    def get_one_person_data(self,
            person: dict[str:int, str: dict[int: np.ndarray[np.float32]]]) -> np.ndarray[Any, dtype[np.float32]]:
        coordinate = person['frame_info']
        denoised_data = np.zeros((list(coordinate.keys())[-1]+1, self.node_num * self.dims), dtype=np.float32)
        for key, value in coordinate.items():
            denoised_data[key, :] = value
        denoised_data = np.concatenate((denoised_data, np.zeros_like(denoised_data)), axis=1)
        return denoised_data

    def get_valid_frames_by_spread(self, person: dict[str:int, str:dict[int, np.ndarray[np.float32]]]) -> list[int]:
        valid_frames = []
        for frame_index, frame_info in person['frame_info'].items():
            x = frame_info[::self.dims]
            y = frame_info[1::self.dims]
            if x.max() - x.min() <= self.NOISE_SPR_THRESHOLD1 * (y.max() - y.min()):
                valid_frames.append(frame_index)
        return valid_frames

    @staticmethod
    def get_motion(frame_info: dict[int, np.ndarray[np.float32]], valid_frame: list[int]) -> np.float32:
        frame_info_copy = frame_info.copy()
        frame_info = [frame_info for frame_index, frame_info in frame_info_copy.items()]
        frame_info = np.array(frame_info, np.float32)
        motion1 = np.sum(np.var(frame_info, axis=0))
        frame_info = [frame_info for frame_index, frame_info in frame_info_copy.items() if frame_index in valid_frame]
        frame_info = np.array(frame_info, np.float32)
        motion2 = np.sum(np.var(frame_info, axis=0))
        return motion1 if motion1 < motion2 else motion2

    def denoising_for_no_overlap(self, persons: list[dict[str: int, str: dict[int: np.ndarray[np.float32]], str: np.float32]]) -> np.ndarray[np.float32]:
        length = max([list(i['frame_info'].keys())[-1] for i in persons]) + 1
        coordinates = np.zeros((length, 2 * self.node_num *  self.dims), dtype=np.float32)
        person1 = persons[0]
        start1, end1 = list(person1['frame_info'].keys())[0], list(person1['frame_info'].keys())[-1]
        coordinates[list(person1['frame_info'].keys()), :self.node_num * self.dims] = np.array(list(person1['frame_info'].values()), np.float32)
        del persons[0]

        start2, end2 = [0, 0]

        while len(persons) > 0:
            coordinate = persons[0]
            start, end = list(coordinate['frame_info'].keys())[0], list(coordinate['frame_info'].keys())[-1]
            if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                coordinates[list(coordinate['frame_info'].keys()), :self.node_num * self.dims] = np.array(list(coordinate['frame_info'].values()), np.float32)
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                coordinates[list(coordinate['frame_info'].keys()), self.node_num * self.dims:] = np.array(list(coordinate['frame_info'].values()), np.float32)
                start2 = min(start, start2)
                end2 = max(end, end2)
            del persons[0]

        return coordinates

    def denoise_and_normalize_data(self, persons: list[dict[str: int, str: dict[int, np.ndarray[np.float32]]]]) -> np.ndarray[Any, dtype[np.float32]] | None:
        if len(persons) > 1:
            candidates = []
            for person in persons:
                if len(person['frame_info']) > self.NOISE_LEN_THRESHOLD:
                    candidates.append(person)
            if len(candidates) == 1:
                return self.get_one_person_data(candidates[0])
            elif len(candidates) == 0:
                return None
            persons = candidates.copy()
            candidates = []
            for person in persons:
                valid_frames = self.get_valid_frames_by_spread(person)
                if len(valid_frames) > self.NOISE_SPR_THRESHOLD2 * len(person['frame_info']):
                    person['motion'] = np.sum(np.var(np.array(list(person['frame_info'].values()), dtype=np.float32).reshape((-1, 3)), axis=0))
                    candidates.append(person)
            if len(candidates) == 1:
                return self.get_one_person_data(candidates[0])
            elif len(candidates) == 0:
                return None
            candidates = sorted(candidates, key=lambda candidate: candidate['motion'], reverse=True)
            denoised_data = self.denoising_for_no_overlap(candidates)
            return denoised_data
        return self.get_one_person_data(persons[0])

    def serialize_data(self, processed_data: np.ndarray[np.float32]) -> \
            ndarray[Any, dtype[np.float32]] | ndarray[Any, dtype[Any]]:
        person1 = processed_data[:, :self.node_num * self.dims]
        person2 = processed_data[:, self.node_num * self.dims:]
        person1_missing_frames = np.where((np.all(person1 == 0, axis=1)))[0]
        person2_missing_frames = np.where((np.all(person2 == 0, axis=1)))[0]
        length = processed_data.shape[0]
        person1_origin_index = np.where(np.all(person1 != 0, axis=1))[0][0]
        origin = person1[person1_origin_index, 3:6]
        serialized_data = processed_data
        serialized_data -= np.tile(origin, (length, self.node_num * 2))
        serialized_data[person1_missing_frames, :self.node_num * self.dims] = 0
        serialized_data[person2_missing_frames, self.node_num * self.dims:] = 0
        return serialized_data

    def read_data(self, filenames: list[str]) -> list[dict[str:int, str:int, str:int, str:int, str:np.ndarray[Any, np.float32]]]:
        if os.path.exists(self.to_be_removed_files):
            with open(self.to_be_removed_files) as file:
                to_be_removed_files = file.readlines()
                to_be_removed_files = [i.strip() + self.file_suffix for i in to_be_removed_files]
                to_be_removed_files = set(to_be_removed_files)
                current_filenames = set(filenames)
                filenames = list(current_filenames - to_be_removed_files)
                removed_files = current_filenames - set(filenames)
                for i in removed_files:
                    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {i} is skipped for missed skeleton.')
        filenames.sort()
        for filename in filenames:
            camera = int(filename[self.camera_start:self.camera_end])
            main_performer = int(filename[self.performer_start:self.performer_end])
            label = int(filename[self.label_start: self.label_end])
            with open(os.path.join(self.raw_dataset_dir, filename), 'r') as file:
                data = file.readlines()
            data = self.read_file(data, self.dims)
            if not data:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {filename} is skipped for length.')
                continue
            denoised_and_normalized_data = self.denoise_and_normalize_data(data)
            if denoised_and_normalized_data is None:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {filename} is skipped for noise.')
                continue
            serialized_data = self.serialize_data(denoised_and_normalized_data)
            self.processed_data.append({'filename': filename, 'camera': camera, 'main_performer': main_performer, 'label': label-1, 'data': serialized_data})
        return self.processed_data

