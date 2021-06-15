for i in `cat log`
do
    for seq in 32 64 128 256
    do
        echo $i $seq
        python run_onnx.py --model $i/$i.onnx --backend cpu --batch 1 --seq $seq
    done
done
