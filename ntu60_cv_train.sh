#!/bin/bash

python ./main.py --config-path ./config/training_config/ntu60_cv_joint.json

python ./main.py --config-path ./config/training_config/ntu60_cv_joint_motion.json

python ./main.py --config-path ./config/training_config/ntu60_cv_bone.json

python ./main.py --config-path ./config/training_config/ntu60_cv_bone_motion.json

