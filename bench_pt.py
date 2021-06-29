#!/usr/bin/env python
import numpy as np
import time
import sys
import argparse
import torch


def benchmark(model_path, batch, seq, backend, N=1):
    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
    }

    if backend == "cpu":
        feed_dict = [
            torch.tensor(np.random.randint(0, 10000, size=[batch,seq]).astype("int64")),
            torch.tensor(np.ones([batch,seq]).astype("int64"))
        ]
        if "distilbert" not in model_path and "roberta" not in model_path:
            shape["token_type_ids"] = (batch, seq)
            feed_dict.append(torch.tensor(np.zeros([batch,seq]).astype("int64")))
    else:
        feed_dict = [
            torch.tensor(np.random.randint(0, 10000, size=[batch,seq]).astype("int64")).cuda(),
            torch.tensor(np.ones([batch,seq]).astype("int64")).cuda()
        ]
        if "distilbert" not in model_path and "roberta" not in model_path:
            shape["token_type_ids"] = (batch, seq)
            feed_dict.append(torch.tensor(np.zeros([batch,seq]).astype("int64")).cuda())

    loaded_model = torch.jit.load(model_path)
    if backend == "gpu":
        loaded_model.to('cuda')
    loaded_model.eval()
    for p in loaded_model.parameters():
        p.requires_grad_(False)

    for _ in range(10):
        res = loaded_model(*feed_dict)

    t1 = time.time()
    for _ in range(N):
        res = loaded_model(*feed_dict)
    t2 = time.time()

    dt = t2 - t1
    inf_time = dt/N*1000
    return inf_time

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--backend", type=str, required=False, default="cpu")
args = parser.parse_args()
model_name = args.model
backend = args.backend

if model_name:
    model_names = [model_name]
else:
    with open("models.txt") as fh:
        model_names = fh.readlines()
        model_names = [model.rstrip() for model in model_names]

batchs = [1, 4]
seqs = [32, 64, 128, 256]
for batch in batchs:
    print("---------------begin profiling PT batch={}------------------".format(batch)) 
    for model_name in model_names:
        model_path = "pt_models/{}/{}.pt".format(model_name, model_name)
        line = "{}".format(model_name, batch)
        for seq in seqs:
            latency = benchmark(model_path, batch, seq, backend, N=100)
            line += ",{}".format(latency)
        print(line)
