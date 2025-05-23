#!/bin/bash

python ./main.py --config-path ./config/training_config/ntu120_csub_joint.json

python ./main.py --config-path ./config/training_config/ntu120_csub_joint_motion.json

python ./main.py --config-path ./config/training_config/ntu120_csub_bone.json

python ./main.py --config-path ./config/training_config/ntu120_csub_bone_motion.json

