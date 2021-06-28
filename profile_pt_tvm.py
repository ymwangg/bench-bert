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
args = parser.parse_args()

model_path = args.model
batch, seq = args.batch, args.seq

shape = {
    "input_ids" : (batch, seq),
    "attention_mask" : (batch, seq),
}

feed_dict = {
    'input_ids' : np.random.randint(0, 10000, size=[batch,seq]).astype("int64"),
    'attention_mask' : np.zeros([batch,seq]).astype("int64"),
}
if "distilbert" not in model_path and "roberta" not in model_path:
    shape["token_type_ids"] = (batch, seq)
    feed_dict["token_type_ids"] = np.zeros([batch,seq]).astype("int64")

model = torch.jit.load(model_path)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
mod, par = relay.frontend.from_pytorch(model, [(k,v) for k,v in shape.items()], default_dtype="float32")


target = 'llvm -mcpu=skylake-avx512 -libs=mkl,mlas'
with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    lib = relay.build(mod, params=par, target=target)

print("done compilation")

ctx = tvm.cpu(0)
m = graph_executor.GraphModule(lib["default"](ctx))
m.run(**feed_dict)
for _ in range(10):
    m.run()
time.sleep(1)
ftimer = m.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=10)
dt = np.mean(ftimer().results)
print("tvm_time = {}".format(dt*1000))
tvm_res = m.get_output(1).asnumpy()
print("tvm_res sum = {}".format(np.sum(tvm_res)))

debug_m = debug_runtime.create(lib.graph_json, lib.lib, ctx)
debug_m.set_input(**feed_dict)
debug_m.run()

if "distilbert" not in model_path and "roberta" not in model_path:
    input = [torch.tensor(feed_dict['input_ids'].astype("int64")), torch.tensor(feed_dict['attention_mask'].astype("int64")), torch.tensor(feed_dict['token_type_ids'].astype("int64"))]
else:
    input = [torch.tensor(feed_dict['input_ids'].astype("int64")), torch.tensor(feed_dict['attention_mask'].astype("int64"))]

N = 100
t1 = time.time()
for _ in range(N):
    pt_res = model(*input)
t2 = time.time()
dt = t2 - t1
print("pt_time = {}".format(dt/N*1000))
print("pt_res sum = {}".format(np.sum(np.array(pt_res[1]))))

def benchmark(model_path, batch, seq, N=1):
    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
    }

    feed_dict = [
        torch.tensor(np.random.randint(0, 10000, size=[batch,seq]).astype("int64")),
        torch.tensor(np.zeros([batch,seq]).astype("int64"))
    ]
    if "distilbert" not in model_path and "roberta" not in model_path:
        shape["token_type_ids"] = (batch, seq)
        feed_dict.append(torch.tensor(np.zeros([batch,seq]).astype("int64")))

    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()
    for p in loaded_model.parameters():
        p.requires_grad_(False)

    # for _ in range(10):
    #     res = loaded_model(*feed_dict)

    t1 = time.time()
    for _ in range(N):
        loaded_model(*feed_dict)
    t2 = time.time()
    dt = t2 - t1
    inf_time = dt/N*1000
    return inf_time
print(model_path, batch, seq)
print(benchmark(model_path, batch, seq, N=100))
