#!/bin/bash
bash compile_tvm_models.sh
python bench_tvm.py
python bench_onnx.py
