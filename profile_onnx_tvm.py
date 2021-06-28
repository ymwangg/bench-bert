#!/usr/bin/env python
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys
import argparse
import onnx

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
args = parser.parse_args()

model_path = args.model
model = onnx.load(model_path)
batch, seq = args.batch, args.seq

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

mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)


target = 'llvm -mcpu=skylake-avx512 -libs=mkl,mlas'
with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    #executable = relay.vm.compile(mod, params = params, target=target)
    lib = relay.build(mod, params = par, target=target)

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
tvm_res = m.get_output(0).asnumpy()
print("tvm_res sum = {}".format(np.sum(tvm_res)))

debug_m = debug_runtime.create(lib.graph_json, lib.lib, ctx)
debug_m.set_input(**feed_dict)
debug_m.run()

import onnxruntime as rt
N = 1000
sess = rt.InferenceSession(model_path)
sess.run(['output_0'], feed_dict)
time.sleep(1)
t1 = time.time()
for _ in range(N):
    onnx_res = sess.run(['output_0'], feed_dict)
t2 = time.time()
print("onnx_time = {}".format((t2 - t1)/N*1000))
print("onnx_res sum = {}".format(np.sum(onnx_res[0])))
