#!/usr/bin/env python
import tvm
from tvm import relay
import onnx
from tvm.contrib import graph_executor
import numpy as np
import time
from tvm.contrib.debugger import debug_runtime
import sys
import argparse

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--backend", type=str, required=False, default="cublas")
args = parser.parse_args()

model_path = args.model
model = onnx.load(model_path)
batch, seq = args.batch, args.seq
backend = args.backend

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

mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)

print("using backend={}".format(backend))
if backend == "cublas":
    target = 'cuda -libs=cublas'
    with relay.build_config(opt_level=3, required_pass=["FastMath"]):
        #executable = relay.vm.compile(mod, params = params, target=target)
        lib = relay.build(mod, params = par, target=target)
elif backend == "trt":
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod, config = partition_for_tensorrt(mod, par)
    target = 'cuda'
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
        #executable = relay.vm.compile(mod, params = params, target=target)
        lib = relay.build(mod, params = par, target=target)
else:
    raise RuntimeError("wrong backend={}".format(backend))
print("done compilation")

ctx = tvm.cuda(0)
m = graph_executor.GraphModule(lib["default"](ctx))
m.run(**feed_dict)
for _ in range(10):
    m.run()
time.sleep(1)
ftimer = m.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=10)
dt = np.mean(ftimer().results)
print("tvm_time({}) = {}".format(backend, dt*1000))
tvm_res = m.get_output(1).asnumpy()
print("tvm_res sum = {}".format(np.sum(tvm_res)))

#debug_m = debug_runtime.create(lib.graph_json, lib.lib, ctx)
#debug_m.set_input(**feed_dict)
#debug_m.run()

import onnxruntime as rt
N = 1000
sess = rt.InferenceSession(model_path)
sess.run(['output_0','output_1'], feed_dict)
time.sleep(1)
t1 = time.time()
for _ in range(N):
    onnx_res = sess.run(['output_0','output_1'], feed_dict)
t2 = time.time()
print("onnx_time = {}".format((t2 - t1)/N*1000))
print("onnx_res sum = {}".format(np.sum(onnx_res[1])))
