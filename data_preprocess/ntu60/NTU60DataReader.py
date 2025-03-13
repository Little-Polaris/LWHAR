from enum import Enum

import numpy as np

from data_preprocess.DataReader import DataReader

class NTU60DataReader(DataReader):
    def __init__(self, config: dict):
        super(NTU60DataReader, self).__init__(config)

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
                coordinate = np.array(coordinate, np.float32).reshape(self.node_num * self.dims)
                if current_person_index == -1:
                    persons.append({'person_id': current_person_id, 'frame_info': {
                        frame_index: coordinate}})
                else:
                    persons[current_person_index]['frame_info'][frame_index] = coordinate
        if len(persons) == 1 and len(persons[0]['frame_info']) < self.invalid_frame_num_lower_bound:
            return None
        return persons