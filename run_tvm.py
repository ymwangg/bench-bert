import tvm
from tvm import relay
import onnx
from tvm.contrib import graph_runtime
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys
import argparse


parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model_prefix", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--N", type=int, default=1000, required=False)

args = parser.parse_args()

prefix = args.model_prefix
batch, seq = args.batch, args.seq
N = args.N

def load_model(prefix):
    lib = tvm.runtime.load_module("{}.so".format(prefix))
    with open("{}.json".format(prefix), 'r') as fh:
        graph = fh.read()
    with open("{}.params".format(prefix), 'rb') as fh:
        params = fh.read()
    return lib, graph, params

shape = {
    "input_ids" : (batch, seq),
    "attention_mask" : (batch, seq),
    "token_type_ids" : (batch, seq)
}

lib0, graph0, params0 = load_model(prefix)

feed_dict = {
    'input_ids' : np.random.randint(0, 10000, size=[batch,seq]).astype("int64"),
    'attention_mask' : np.zeros([batch,seq]).astype("int64"),
    'token_type_ids' : np.zeros([batch,seq]).astype("int64")
}


ctx = tvm.cpu()
m0 = graph_runtime.graph_executor.create(graph0, lib0, ctx)
#m.set_input("input_ids", np.random.randint(0, 30522, size=[1,64]).astype("int32"))
m0.load_params(params0)
m0.set_input(**feed_dict)

m0.run()
dt = 0.0
for _ in range(N):
    feed_dict = {
        'input_ids' : np.random.randint(0, 3, size=[batch,seq]).astype("int64"),
        'attention_mask' : np.zeros([batch,seq]).astype("int64"),
        'token_type_ids' : np.zeros([batch,seq]).astype("int64")
    }
    m0.set_input(**feed_dict)
    t1 = time.time()
    m0.run()
    t2 = time.time()
    dt += t2 - t1
print(dt/N*1000)

#debug_m0 = debug_runtime.create(graph0, lib0, ctx)
#debug_m0.load_params(params0)
#debug_m0.run()

