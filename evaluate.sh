#!/bin/bash
for i in `cat models.txt`
do
    for batch in 1 4
    do
        for seq in 32 64 128 256
        do
	    echo "batch=$batch seq=$seq"
            #python compile_tvm_model.py --model models/$i/$i.onnx --batch $batch --seq $seq --target "cuda"
	    python bench_single_tvm.py --model $i --batch $batch --seq $seq --target "cuda"
        done
    done
done
