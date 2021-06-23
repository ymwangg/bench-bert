#!/bin/bash
i=$1
for batch in 1 4
do
    for seq in 32 64 128 256
    do
        python compile_tvm_model.py --model $i --batch $batch --seq $seq --target "$2" --model_type onnx
    done
done
