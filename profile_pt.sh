#!/bin/bash
export KMP_AFFINITY=verbose,scatter
#./compile_all_pt_models.sh "llvm -mcpu=skylake-avx512 -libs=mkl,mlas"
./compile_all_pt_models.sh "cuda -libs=cublas"
python bench_tvm.py --device cpu --model_type pt
python bench_tvm.py --device cpu --model_type pt
python bench_pt.py
python bench_pt.py
