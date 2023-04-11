#!/bin/bash

python3 apples_detection/predict.py \
    data.patches_suffix="-patches_inst4_ker400max" \
    data.batch_size=8 \
    trainer=gpu \
    trainer.devices=[1]
