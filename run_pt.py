#!/usr/bin/env python
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys
import argparse
import torch

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--backend", type=str, required=False, default="cpu")
args = parser.parse_args()

model_name = args.model
model_path = "pt_models/{}/{}.pt".format(model_name, model_name)
batch, seq = args.batch, args.seq
backend = args.backend

shape = {
    "input_ids" : (batch, seq),
    "attention_mask" : (batch, seq),
}

feed_dict = {
    'input_ids' : np.random.randint(0, 10000, size=[batch,seq]).astype("int64"),
    'attention_mask' : np.ones([batch,seq]).astype("int64"),
}
if "distilbert" not in model_path and "roberta" not in model_path:
    shape["token_type_ids"] = (batch, seq)
    feed_dict["token_type_ids"] = np.zeros([batch,seq]).astype("int64")

model = torch.jit.load(model_path)
if backend == "gpu":
    model.to("cuda")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if backend == "cpu":
    if "distilbert" not in model_path and "roberta" not in model_path:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")), torch.tensor(feed_dict['attention_mask'].astype("int64")), torch.tensor(feed_dict['token_type_ids'].astype("int64"))]
    else:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")), torch.tensor(feed_dict['attention_mask'].astype("int64"))]
else:
    if "distilbert" not in model_path and "roberta" not in model_path:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")).cuda(), torch.tensor(feed_dict['attention_mask'].astype("int64")).cuda(), torch.tensor(feed_dict['token_type_ids'].astype("int64")).cuda()]
    else:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")).cuda(), torch.tensor(feed_dict['attention_mask'].astype("int64")).cuda()]

#for _ in range(10):
#    pt_res = model(*feed_dict)

N = 1000
t1 = time.time()
for _ in range(N):
    pt_res = model(*feed_dict)
t2 = time.time()
dt = t2 - t1

print("pt_time = {}".format(dt/N*1000))
if backend == "cpu":
    pt_res = np.array(pt_res[0])
else:
    pt_res = np.array(pt_res[0].cpu())
print("pt_res sum = {}".format(np.sum(pt_res)))
