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
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--backend", type=str, required=False, default="cublas")
parser.add_argument("--type", type=str, required=True)
args = parser.parse_args()

model_name = args.model
batch, seq = args.batch, args.seq
backend = args.backend
model_type = args.type

if model_type == "onnx":
    model_path = "models/{}/{}.onnx".format(model_name, model_name)
else:
    model_path = "pt_models/{}/{}.pt".format(model_name, model_name)

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

if model_type == "onnx":
    model = onnx.load(model_path)
    mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)
else:
    import torch
    model = torch.jit.load(model_path)
    model.to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mod, par = relay.frontend.from_pytorch(model, [(k,v) for k,v in shape.items()], default_dtype="float32")
#print(mod)
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_expr, rewrite, is_op, is_constant
class LayerNormRewrite(DFPatternCallback):
    def __init__(self):
        super(LayerNormRewrite, self).__init__()
        self.x = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.epsilon = wildcard()
        mean = is_op("mean")
        power = is_op("power")
        sqrt = is_op("sqrt")
        expr_a = self.x - mean(self.x)
        expr_b = sqrt(mean(power(expr_a, is_expr(relay.const(2.0)))) + self.epsilon)
        expr_c = expr_a / expr_b * self.gamma + self.beta
        self.pattern = expr_c

    def callback(self, pre, post, node_map):
        new_x = node_map[self.x][0]
        new_gamma = node_map[self.gamma][0]
        new_beta = node_map[self.beta][0]
        new_epsilon = float(node_map[self.epsilon][0].data.asnumpy())
        return relay.nn.layer_norm(new_x, new_gamma, new_beta, epsilon=new_epsilon)
#expr = rewrite(LayerNormRewrite(), mod['main'])
#mod = tvm.IRModule.from_expr(expr)
#print(mod)
#import pdb
#pdb.set_trace()

class BatchMatmulRewriteRule1(DFPatternCallback):
    def __init__(self):
        super(BatchMatmulRewriteRule1, self).__init__()
        self.A = wildcard()
        self.B = wildcard()
        transpose = is_op("transpose")
        batch_matmul = is_op("nn.batch_matmul")
        expr = batch_matmul(self.A, transpose(self.B).has_attr({"axes":[0, 2, 1]}))
        self.pattern = expr

    def callback(self, pre, post, node_map):
        A = node_map[self.A][0]
        B = node_map[self.B][0]
        return relay.op.neo_batch_matmul(A, B, False)

class BatchMatmulRewriteRule2(DFPatternCallback):
    def __init__(self):
        super(BatchMatmulRewriteRule2, self).__init__()
        self.A = wildcard()
        self.B = is_constant()
        batch_matmul = is_op("nn.batch_matmul")
        expr = batch_matmul(self.A, self.B)
        self.pattern = expr

    def callback(self, pre, post, node_map):
        A = node_map[self.A][0]
        B = node_map[self.B][0]
        return relay.op.neo_batch_matmul(A, relay.transpose(B, axes=[0, 2, 1]), False)

#expr = mod['main']
#expr = rewrite(BatchMatmulRewriteRule1(), expr)
#expr = rewrite(BatchMatmulRewriteRule2(), expr)
#mod = tvm.IRModule.from_expr(expr)
#print(mod)

print("using backend={}".format(backend))
if backend == "cublas":
    target = 'cuda -libs=cublas'
    with relay.build_config(opt_level=3, required_pass=["FastMath"]):
        lib = relay.build(mod, params = par, target=target)
elif backend == "trt":
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    #mod, config = partition_for_tensorrt(mod, par, use_implicit_batch=True)
    mod, config = partition_for_tensorrt(mod, par, use_implicit_batch=False, use_neo_batch_matmul=False)
    target = 'cuda -libs=cublas'
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
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
ftimer = m.module.time_evaluator("run", ctx, min_repeat_ms=50, repeat=100)
dt = np.mean(ftimer().results)
print("tvm_time({}) = {}".format(backend, dt*1000))
tvm_res = m.get_output(1).asnumpy()
print("tvm_res sum = {}".format(np.sum(tvm_res)))

#debug_m = debug_runtime.create(lib.graph_json, lib.lib, ctx)
#debug_m.set_input(**feed_dict)
#debug_m.run()

if model_type == "onnx":
    import onnxruntime as rt
    N = 1000
    sess = rt.InferenceSession(model_path)
    output_names = [out.name for out in model.graph.output]
    sess.run(output_names, feed_dict)
    time.sleep(1)
    t1 = time.time()
    for _ in range(N):
        onnx_res = sess.run([], feed_dict)
    t2 = time.time()
    print("onnx_time = {}".format((t2 - t1)/N*1000))
    print("onnx_res sum = {}".format(np.sum(onnx_res[1])))
else:
    if "distilbert" not in model_path and "roberta" not in model_path:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")).cuda(), torch.tensor(feed_dict['attention_mask'].astype("int64")).cuda(), torch.tensor(feed_dict['token_type_ids'].astype("int64")).cuda()]
    else:
        feed_dict = [torch.tensor(feed_dict['input_ids'].astype("int64")).cuda(), torch.tensor(feed_dict['attention_mask'].astype("int64")).cuda()]

    for _ in range(10):
        pt_res = model(*feed_dict)

    N = 1000
    t1 = time.time()
    for _ in range(N):
        pt_res = model(*feed_dict)
    t2 = time.time()
    dt = t2 - t1

    print("pt_time = {}".format(dt/N*1000))
    pt_res = np.array(pt_res[1].cpu())
    print("pt_res sum = {}".format(np.sum(pt_res)))
