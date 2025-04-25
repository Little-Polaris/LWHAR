#!/bin/bash

python ./main.py --config-path ./config/training_config/ntu60_cs_joint.json

python ./main.py --config-path ./config/training_config/ntu60_cs_joint_motion.json

python ./main.py --config-path ./config/training_config/ntu60_cs_bone.json

python ./main.py --config-path ./config/training_config/ntu60_cs_bone_motion.json

