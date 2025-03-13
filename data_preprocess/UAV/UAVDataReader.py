from enum import Enum

import numpy as np

from data_preprocess.DataReader import DataReader

class UAVDataReader(DataReader):
    def __init__(self, config: dict):
        super(UAVDataReader, self).__init__(config)

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
                coordinate = np.array(coordinate, np.float32)
                if False in coordinate.any(axis=1) or np.nan in coordinate:
                    continue
                if len(persons) > i:
                    persons[i]['frame_info'][frame_index] = coordinate
                else:
                    persons.append({'frame_info': {frame_index: coordinate}})
        if len(persons) == 1 and len(persons[0]['frame_info']) < self.invalid_frame_num_lower_bound:
            return None
        return persons
