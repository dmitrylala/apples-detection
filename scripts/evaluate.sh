#!/bin/bash

python3 apples_detection/eval.py \
    model_name="best_faster_rcnn_09.04.23" \
    trainer=gpu \
    trainer.devices=2 \
    data.batch_size=4
