import os
from datetime import datetime
from enum import Enum

import numpy as np

from data_preprocess.DataReader import DataReader

class NTU120DataReader(DataReader):
    def __init__(self, config: dict):
        super(NTU120DataReader, self).__init__(config)
        self.camera_start = config['camera_start']
        self.camera_end = config['camera_end']
        self.center_joint = config['center_joint']

    class Joint(Enum):
        BASE_OF_SPINE = 0
        MIDDLE_OF_SPINE = 1
        NECK = 2
        HEAD = 3
        LEFT_SHOULDER = 4
        LEFT_ELBOW = 5
        LEFT_WRIST = 6
        LEFT_HAND = 7
        RIGHT_SHOULDER = 8
        RIGHT_ELBOW = 9
        RIGHT_WRIST = 10
        RIGHT_HAND = 11
        LEFT_HIP = 12
        LEFT_KNEE = 13
        LEFT_ANKLE = 14
        LEFT_FOOT = 15
        RIGHT_HIP = 16
        RIGHT_KNEE = 17
        RIGHT_ANKLE = 18
        RIGHT_FOOT = 19
        SPINE = 20
        TIP_OF_LEFT_HAND = 21
        LEFT_THUMB = 22
        TIP_OF_RIGHT_HAND = 23
        RIGHT_THUMB = 24

    def read_file(self, file_content: list[str], dims: int):
        current_row = 0
        row_num = len(file_content)
        current_row += 1
        persons = []
        frame_index = -1
        while current_row < row_num:
            frame_index += 1
            person_num = int(file_content[current_row].strip())
            if person_num == 0:
                while current_row < row_num and file_content[current_row].strip() == '0':
                    current_row += 1
                if current_row == row_num:
                    current_row += 1
                    continue
                person_num = int(file_content[current_row].strip())
            current_row += 1
            for i in range(person_num):
                current_person_id = int(file_content[current_row].split()[0])
                current_row += 1
                node_num = int(file_content[current_row].strip())
                current_row += 1
                coordinate = []
                for j in range(node_num):
                    node = file_content[current_row].split()[:dims]
                    current_row += 1
                    coordinate.append(node)
                current_person_index = -1
                for j in range(len(persons)):
                    if persons[j]['person_id'] == current_person_id:
                        current_person_index = j
                        break
                coordinate = np.array(coordinate, np.float32).reshape(self.num_joint * self.dims)
                if current_person_index == -1:
                    persons.append({'person_id': current_person_id, 'frame_info': {
                        frame_index: coordinate}})
                else:
                    persons[current_person_index]['frame_info'][frame_index] = coordinate
        if len(persons) == 1 and len(persons[0]['frame_info']) < self.invalid_frame_num_lower_bound:
            return None
        return persons

    def get_valid_frames_by_spread(self, person: dict[str:int, str:dict[int, np.ndarray[np.float32]]]) -> list[int]:
        valid_frames = []
        for frame_index, frame_info in person['frame_info'].items():
            x = frame_info[::self.dims]
            y = frame_info[1::self.dims]
            if x.max() - x.min() <= self.NOISE_SPR_THRESHOLD1 * (y.max() - y.min()):
                valid_frames.append(frame_index)
        return valid_frames

    def serialize_data(self, processed_data):
        person1 = processed_data[:, :self.num_joint * self.dims]
        person2 = processed_data[:, self.num_joint * self.dims:]
        person1_missing_frames = np.where((np.all(person1 == 0, axis=1)))[0]
        person2_missing_frames = np.where((np.all(person2 == 0, axis=1)))[0]
        length = processed_data.shape[0]
        person1_origin_index = np.where(np.all(person1 != 0, axis=1))[0][0]
        origin = person1[person1_origin_index, (self.center_joint-1)*self.dims:(self.center_joint-1)*self.dims+self.dims]
        serialized_data = processed_data
        serialized_data -= np.tile(origin, (length, self.num_joint * 2))
        serialized_data[person1_missing_frames, :self.num_joint * self.dims] = 0
        serialized_data[person2_missing_frames, self.num_joint * self.dims:] = 0
        return serialized_data

    def read_data(self, filenames):
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
            denoised_data = self.denoise_data(data)
            if denoised_data is None:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {filename} is skipped for noise.')
                continue
            serialized_data = self.serialize_data(denoised_data)
            self.processed_data.append({'filename': filename, 'camera': camera, 'main_performer': main_performer, 'label': label-1, 'data': serialized_data})
        return self.processed_data