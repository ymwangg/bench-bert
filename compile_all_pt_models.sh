#!/bin/bash
$target=$1
for i in `cat models.txt`
do
    for batch in 1 4
    do
        for seq in 32 64 128 256
        do
            python compile_tvm_model.py --model $i --batch $batch --seq $seq --target $target --model_type pt
        done
    done
done
