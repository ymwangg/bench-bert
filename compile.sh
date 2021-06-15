for i in `cat models.txt`
do
    for seq in 32 64 128 256
    do
        python compile.py --model $i/$i.onnx --batch 1 --seq $seq
    done
done
