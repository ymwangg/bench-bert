#!/bin/bash
mkdir $1
cd $1
python ../convert_graph_to_onnx.py --framework pt --model $1 $1.onnx
cd ..
