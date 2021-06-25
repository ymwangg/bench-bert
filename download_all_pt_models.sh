#!/bin/bash
if [ -d "models" ]; then
    rm -rf ./pt_models
fi
mkdir pt_models
for i in `cat models.txt`
do
    ./download_pt_model.sh $i
done
