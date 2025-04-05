import os
from datetime import datetime
from enum import Enum

import numpy as np

from data_preprocess.DataReader import DataReader

class UAVDataReader(DataReader):
    def __init__(self, config: dict):
        super(UAVDataReader, self).__init__(config)
        self.center_joint = config['center_joint']

    class Joint(Enum):
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_ELBOW = 7
        RIGHT_ELBOW = 8
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16

        def __mul__(self, other):
            if isinstance(other, UAVDataReader.Joint):
                return self.value * other.value
            elif isinstance(other, (int, float)):
                return self.value * other
            else:
                return NotImplemented  # 让 Python 尝试其他对象的 __rmul__

    def read_file(self, file_content: list[str], dims: int):
        current_row = 0
        row_num = len(file_content)
        current_row += 1
        persons = []
        frame_index = -1
        while current_row < row_num:
            frame_index += 1
            person_num = int(file_content[current_row].strip())
            current_row += 1
            for i in range(person_num):
                current_row += 1
                node_num = int(file_content[current_row].strip())
                current_row += 1
                coordinate = []
                for j in range(node_num):
                    node = file_content[current_row].split()[:dims]
                    current_row += 1
                    coordinate.append(node)
                coordinate = np.array(coordinate, np.float32)#.reshape(self.num_joint * self.dims)
                if np.sum(coordinate.any(axis=1)) < self.num_joint - 1 or np.nan in coordinate:
                    continue
                coordinate = coordinate.reshape(self.num_joint * self.dims)
                if len(persons) > i:
                    persons[i]['frame_info'][frame_index] = coordinate
                else:
                    persons.append({'frame_info': {frame_index: coordinate}})

        persons = [i for i in persons if len(i['frame_info']) >= list(i['frame_info'].keys())[-1] // 2]

        if len(persons) == 1 and len(persons[0]['frame_info']) < self.invalid_frame_num_lower_bound:
            return None
        return persons

    def get_valid_frames_by_spread(self, person: dict[str:int, str:dict[int, np.ndarray[np.float32]]]) -> list[int]:
        return [i for i in person['frame_info'].keys()]

    def serialize_data(self, processed_data):
        person1 = processed_data[:, :self.num_joint * self.dims]
        person2 = processed_data[:, self.num_joint * self.dims:]
        person1_missing_frames = np.where((np.all(person1 == 0, axis=1)))[0]
        person2_missing_frames = np.where((np.all(person2 == 0, axis=1)))[0]
        length = processed_data.shape[0]
        person1_origin_index = np.where(np.all(person1 != 0, axis=1))[0][0]
        hip_left = person1[person1_origin_index, self.Joint.LEFT_HIP * self.dims:self.Joint.LEFT_HIP * self.dims + self.dims]
        hip_right = person1[person1_origin_index, self.Joint.RIGHT_HIP * self.dims:self.Joint.RIGHT_HIP * self.dims + self.dims]
        shoulder_left = person1[person1_origin_index, self.Joint.LEFT_SHOULDER * self.dims:self.Joint.LEFT_SHOULDER * self.dims + self.dims]
        shoulder_right = person1[person1_origin_index, self.Joint.RIGHT_SHOULDER * self.dims:self.Joint.RIGHT_SHOULDER * self.dims + self.dims]
        origin = ((hip_left + hip_right) / 2 + (shoulder_left + shoulder_right) / 2) / 2
        serialized_data = processed_data
        serialized_data -= np.tile(origin, (length, self.num_joint * 2))
        serialized_data[person1_missing_frames, :self.num_joint * self.dims] = 0
        serialized_data[person2_missing_frames, self.num_joint * self.dims:] = 0
        return serialized_data

    def read_data(self, filenames):
        filenames.sort()
        for filename in filenames:
            main_performer = int(filename[self.performer_start:self.performer_end])
            label = int(filename[self.label_start: self.label_end])
            with open(os.path.join(self.raw_dataset_dir, filename), 'r') as file:
                data = file.readlines()
            data = self.read_file(data, self.dims)
            if not data:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {filename} is skipped for length.')
                continue
            denoised_data = self.denoise_data(data)
            if denoised_data is None:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {filename} is skipped for noise.')
                continue
            serialized_data = self.serialize_data(denoised_data)
            self.processed_data.append({'filename': filename, 'main_performer': main_performer, 'label': label, 'data': serialized_data})
        return self.processed_data