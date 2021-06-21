#!/bin/bash
ROOTDIR=`pwd`
if [ -d models/$1 ]; then
    cd models
    rm -rf $1
    cd ..
fi
mkdir models/$1
cd models/$1
python ${ROOTDIR}/convert_graph_to_onnx.py --framework pt --model $1 $1.onnx
cd ${ROOTDIR}
