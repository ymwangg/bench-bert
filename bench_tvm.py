import tvm
from tvm import relay
import onnx
from tvm.contrib import graph_runtime
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys


def load_model(prefix):
    lib = tvm.runtime.load_module("{}.so".format(prefix))
    with open("{}.json".format(prefix), 'r') as fh:
        graph = fh.read()
    with open("{}.params".format(prefix), 'rb') as fh:
        params = fh.read()
    return lib, graph, params


def benchmark(prefix, batch, seq, N=1):
    lib0, graph0, params0 = load_model(prefix)

    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
    }

    feed_dict = {
        'input_ids' : np.random.randint(0, 10000, size=[batch,seq]).astype("int64"),
        'attention_mask' : np.zeros([batch,seq]).astype("int64"),
    }
    if "distilbert" not in prefix and "roberta" not in prefix:
        shape["token_type_ids"] = (batch, seq)
        feed_dict["token_type_ids"] = np.zeros([batch,seq]).astype("int64")


    ctx = tvm.cpu()
    m0 = graph_runtime.graph_executor.create(graph0, lib0, ctx)
    m0.load_params(params0)
    m0.set_input(**feed_dict)

    for _ in range(10):
        m0.run()

    dt = 0.0
    for _ in range(N):
        feed_dict = {
            'input_ids' : np.random.randint(0, 1000, size=[batch,seq]).astype("int64"),
            'attention_mask' : np.zeros([batch,seq]).astype("int64"),
        }
        if "distilbert" not in prefix and "roberta" not in prefix:
            feed_dict['token_type_ids'] = np.zeros([batch,seq]).astype("int64")
        m0.set_input(**feed_dict)
        t1 = time.time()
        m0.run()
        t2 = time.time()
        dt += t2 - t1
    inf_time = dt/N*1000
    return inf_time

with open("models.txt") as fh:
    model_names = fh.readlines()
    model_names = [model.rstrip() for model in model_names]

batchs = [1, 4, 64]
seqs = [32, 64, 128, 256]
for batch in batchs:
    print("---------------begin profiling tvm batch={}------------------".format(batch)) 
    for model_name in model_names:
        line = "{}".format(model_name)
        for seq in seqs:
            model_prefix = "{}/{}-{}-{}".format(model_name, model_name, batch, seq)
            latency = benchmark(model_prefix, batch, seq, N=1000)
            line += ",{}".format(latency)
        print(line)
