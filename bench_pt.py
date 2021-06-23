#!/usr/bin/env python
import numpy as np
import time
import sys
import argparse
import torch


def benchmark(model_path, batch, seq, N=1):
    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
    }

    feed_dict = {
        'input_ids' : torch.tensor(np.random.randint(0, 10000, size=[batch,seq]).astype("int64")),
        'attention_mask' : torch.tensor(np.zeros([batch,seq]).astype("int64")),
    }
    if "distilbert" not in model_path and "roberta" not in model_path:
        shape["token_type_ids"] = (batch, seq)
        feed_dict["token_type_ids"] = torch.tensor(np.zeros([batch,seq]).astype("int64"))

    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()
    for p in loaded_model.parameters():
        p.requires_grad_(False)

    for _ in range(10):
        res = loaded_model(feed_dict['input_ids'], feed_dict['attention_mask'], feed_dict['token_type_ids'])

    dt = 0.0
    t1 = time.time()
    for _ in range(N):
        res = loaded_model(feed_dict['input_ids'], feed_dict['attention_mask'], feed_dict['token_type_ids'])
    t2 = time.time()
    dt += t2 - t1
    inf_time = dt/N*1000
    return inf_time

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=False)
args = parser.parse_args()
model_name = args.model
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
            latency = benchmark(model_path, batch, seq, N=100)
            line += ",{}".format(latency)
        print(line)
