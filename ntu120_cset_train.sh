#!/bin/bash

python ./main.py --config-path ./config/training_config/ntu120_cset_joint.json

python ./main.py --config-path ./config/training_config/ntu120_cset_joint_motion.json

python ./main.py --config-path ./config/training_config/ntu120_cset_bone.json

python ./main.py --config-path ./config/training_config/ntu120_cset_bone_motion.json

