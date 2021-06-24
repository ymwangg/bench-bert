#!/bin/bash
export KMP_AFFINITY=verbose,scatter
python bench_tvm.py
python bench_onnx.py
