#!/bin/bash
mkdir -p pt_models/$1
python ./download_pt_model.py --model $1
