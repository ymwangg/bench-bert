#!/bin/bash
if [ -d "models" ]; then
    rm -rf ./models
fi
mkdir models
for i in `cat models.txt`
do
    ./download_onnx_model.sh $i
done
