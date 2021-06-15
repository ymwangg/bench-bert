#!/bin/bash
bash compile_tvm_models.sh "llvm -mcpu=skylake-avx512 -libs=mkl,mlas"
python bench_tvm.py
python bench_onnx.py
