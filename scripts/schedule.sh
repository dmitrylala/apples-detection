#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python3 apples_detection/train.py \
    experiment=minneapple-maskrcnn-base \
    trainer=gpu \
    trainer.devices=[1]
