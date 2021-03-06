#!/usr/bin/env python
import onnx
import numpy as np
import time
import sys
import argparse
import onnxruntime as rt


def benchmark(model_path, batch, seq, N=1, no_packing=False):
    shape = {
        "input_ids": (batch, seq),
        "attention_mask": (batch, seq),
    }

    feed_dict = {
        "input_ids": np.random.randint(0, 10000, size=[batch, seq]).astype("int64"),
        "attention_mask": np.ones([batch, seq]).astype("int64"),
    }
    if "distilbert" not in model_path and "roberta" not in model_path:
        shape["token_type_ids"] = (batch, seq)
        feed_dict["token_type_ids"] = np.zeros([batch, seq]).astype("int64")

    options = rt.SessionOptions()
    if no_packing:
        options.add_session_config_entry("session.disable_prepacking", "1")
    sess = rt.InferenceSession(model_path, options)

    model = onnx.load(model_path)
    output_names = [out.name for out in model.graph.output]
    for _ in range(10):
        sess.run(output_names, feed_dict)

    dt = 0.0
    t1 = time.time()
    for _ in range(N):
        res = sess.run(output_names, feed_dict)
    t2 = time.time()
    dt += t2 - t1
    inf_time = dt / N * 1000
    return inf_time


def main():
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--batch", type=int, required=False)
    parser.add_argument("--seq", type=int, required=False)
    parser.add_argument("--N", type=int, required=False, default=100)
    args = parser.parse_args()
    model_name = args.model
    batch, seq = args.batch, args.seq
    N = args.N

    if model_name:
        model_names = [model_name]
    else:
        with open("models.txt") as fh:
            model_names = fh.readlines()
            model_names = [model.rstrip() for model in model_names]

    batchs = [batch] if batch else [1, 4]
    seqs = [seq] if seq else [32, 64, 128, 256]

    for batch in batchs:
        print(
            "---------------begin profiling onnx batch={}------------------".format(
                batch
            )
        )
        for model_name in model_names:
            model_path = "models/{}/{}.onnx".format(model_name, model_name)
            line = "{}".format(model_name, batch)
            for seq in seqs:
                latency = benchmark(model_path, batch, seq, N=N, no_packing=False)
                line += ",{}".format(latency)
            print(line)


if __name__ == "__main__":
    main()
