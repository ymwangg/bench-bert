#!/bin/bash
export KMP_AFFINITY=scatter
python bench_tvm.py
python bench_onnx.py
