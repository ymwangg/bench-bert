#!/usr/bin/env python
import sys
import tvm
from tvm import relay
import onnx
import argparse
from tvm.relay.op.contrib import tensorrt
import torch

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--target", type=str, default='llvm -mcpu=skylake-avx512 -libs=mkl,mlas')
parser.add_argument("--use_trt", type=bool, required=False, default=False)
parser.add_argument("--model_type", type=str, default='onnx', required=False)

args = parser.parse_args()
model_name = args.model
batch, seq = args.batch, args.seq
target = args.target
use_trt = args.use_trt
model_type = args.model_type

if model_type == "onnx":
    model_path = "models/{}/{}.onnx".format(model_name, model_name)
    prefix = "models/{}/{}".format(model_name, model_name)
else:
    model_path = "pt_models/{}/{}.pt".format(model_name, model_name)
    prefix = "pt_models/{}/{}".format(model_name, model_name)
print("--------------------Compiling {}--------------------".format(model_path))
print("model_type = {}".format(model_type))
print("target = {}".format(target))
print("use_trt = {}".format(use_trt))

print(prefix)

if "distilbert" in model_path or "roberta" in model_path:
    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
    }
else:
    shape = {
        "input_ids" : (batch, seq),
        "attention_mask" : (batch, seq),
        "token_type_ids" : (batch, seq)
    }

def save_model(graph, lib, params, prefix):
    prefix = "{}-{}-{}".format(prefix, batch, seq)
    lib.export_library("{}.so".format(prefix))
    with open("{}.json".format(prefix), 'w') as fh:
        fh.write(graph)
    with open("{}.params".format(prefix), 'wb') as fh:
        fh.write(relay.save_param_dict(params))

if model_type == "onnx":
    model = onnx.load(model_path)
    mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)
elif model_type == "pt" or model_type == "pytorch":
    model = torch.jit.load(model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mod, par = relay.frontend.from_pytorch(model, [(k,v) for k,v in shape.items()], default_dtype="float32")

if use_trt:
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod, config = partition_for_tensorrt(mod, par, use_implicit_batch=False)
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
        graph, lib, params = relay.build(mod, params=par, target=target)
else:
    with relay.build_config(opt_level=3, required_pass=["FastMath"]):
        graph, lib, params = relay.build(mod, params=par, target=target)

save_model(graph, lib, params, prefix)
