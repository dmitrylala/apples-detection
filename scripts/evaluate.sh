#!/bin/bash

python3 apples_detection/eval.py \
    -m model_name=patches_ker400_min4_epoch11,patches_ker400_min4_epoch28 \
    data.patches_suffix="-patches_inst4_ker400max" \
    trainer=gpu \
    trainer.devices=2 \
    data.batch_size=4
