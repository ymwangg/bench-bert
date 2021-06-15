#!/bin/bash
for i in `cat models.txt`
do
    for batch in 1 4 64
    do
        for seq in 32 64 128 256
        do
            python compile_tvm_model.py --model $i/$i.onnx --batch $batch --seq $seq --target $1
        done
    done
done
