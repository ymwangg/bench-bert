#!/usr/bin/env python
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys
import argparse


def load_model(prefix):
    lib = tvm.runtime.load_module("{}.so".format(prefix))
    with open("{}.json".format(prefix), "r") as fh:
        graph = fh.read()
    with open("{}.params".format(prefix), "rb") as fh:
        params = fh.read()
    return lib, graph, params


def benchmark(prefix, batch, seq, device, N=1):
    lib0, graph0, params0 = load_model(prefix)

    shape = {
        "input_ids": (batch, seq),
        "attention_mask": (batch, seq),
    }

    feed_dict = {
        "input_ids": np.random.randint(0, 10000, size=[batch, seq]).astype("int64"),
        "attention_mask": np.ones([batch, seq]).astype("int64"),
    }
    if "distilbert" not in prefix and "roberta" not in prefix:
        shape["token_type_ids"] = (batch, seq)
        feed_dict["token_type_ids"] = np.zeros([batch, seq]).astype("int64")

    if device == "cpu":
        ctx = tvm.cpu(0)
    elif device == "gpu":
        ctx = tvm.cuda(0)
    else:
        raise RuntimeError("Unknown device={}".format(device))

    m0 = graph_runtime.graph_executor.create(graph0, lib0, ctx)
    m0.load_params(params0)
    m0.set_input(**feed_dict)

    for _ in range(10):
        m0.run()

    # t1 = time.time()
    # for _ in range(N):
    #     m0.run()
    # t2 = time.time()
    # dt = t2 - t1

    ftimer = m0.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=10)
    dt = np.mean(ftimer().results)
    inf_time = dt * 1000
    return inf_time


parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=False)
parser.add_argument(
    "--backend", type=str, required=False, default="cpu", choices=["cpu", "gpu"]
)
parser.add_argument("--batch", type=int, required=False)
parser.add_argument("--seq", type=int, required=False)
parser.add_argument("--N", type=int, required=False, default=100)
parser.add_argument("--type", type=str, required=True, choices=["onnx", "pt"])

args = parser.parse_args()
model_name = args.model
batch, seq = args.batch, args.seq
backend = args.backend
N = args.N
model_type = args.type

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
        "---------------begin profiling {}-tvm batch={}------------------".format(
            model_type, batch
        )
    )
    for model_name in model_names:
        line = "{}".format(model_name)
        for seq in seqs:
            if model_type == "onnx":
                model_prefix = "models/{}/{}-{}-{}".format(
                    model_name, model_name, batch, seq
                )
            else:
                model_prefix = "pt_models/{}/{}-{}-{}".format(
                    model_name, model_name, batch, seq
                )
            latency = benchmark(model_prefix, batch, seq, backend, N=N)
            line += ",{}".format(latency)
        print(line)
