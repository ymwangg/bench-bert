#!/usr/bin/env python
import sys
import tvm
from tvm import relay
import onnx
import argparse
from tvm.relay.op.contrib import tensorrt
import torch


def save_model(graph, lib, params, prefix, batch, seq):
    prefix = "{}-{}-{}".format(prefix, batch, seq)
    lib.export_library("{}.so".format(prefix))
    with open("{}.json".format(prefix), "w") as fh:
        fh.write(graph)
    with open("{}.params".format(prefix), "wb") as fh:
        fh.write(relay.save_param_dict(params))


def compile(model_name, batch, seq, target, use_trt, model_type):

    if model_type == "onnx":
        model_path = "models/{}/{}.onnx".format(model_name, model_name)
        prefix = "models/{}/{}".format(model_name, model_name)
    else:
        model_path = "pt_models/{}/{}.pt".format(model_name, model_name)
        prefix = "pt_models/{}/{}".format(model_name, model_name)

    print("--------------------Compiling {}--------------------".format(model_path))
    print("model_type = {}".format(model_type))
    print("batch, seq = {}, {}".format(batch, seq))
    print("target = {}".format(target))
    print("use_trt = {}".format(use_trt))

    if "distilbert" in model_path or "roberta" in model_path:
        shape = {
            "input_ids": (batch, seq),
            "attention_mask": (batch, seq),
        }
    else:
        shape = {
            "input_ids": (batch, seq),
            "attention_mask": (batch, seq),
            "token_type_ids": (batch, seq),
        }

    if model_type == "onnx":
        model = onnx.load(model_path)
        mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)
    elif model_type == "pt":
        model = torch.jit.load(model_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        mod, par = relay.frontend.from_pytorch(
            model, [(k, v) for k, v in shape.items()], default_dtype="float32"
        )

    if use_trt:
        from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

        mod, config = partition_for_tensorrt(mod, par, use_implicit_batch=False)
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.ext.tensorrt.options": config}
        ):
            graph, lib, params = relay.build(mod, params=par, target=target)
    else:
        with relay.build_config(opt_level=3, required_pass=["FastMath"]):
            graph, lib, params = relay.build(mod, params=par, target=target)

    save_model(graph, lib, params, prefix, batch, seq)


def main():
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--batch", type=int, required=False)
    parser.add_argument("--seq", type=int, required=False)
    parser.add_argument(
        "--target", type=str, default="llvm -mcpu=skylake-avx512 -libs=mkl,mlas"
    )
    parser.add_argument("--use_trt", type=bool, required=False, default=False)
    parser.add_argument(
        "--type", type=str, default="onnx", choices=["onnx", "pt"], required=True
    )

    args = parser.parse_args()
    model_name = args.model
    batch, seq = args.batch, args.seq
    target = args.target
    use_trt = args.use_trt
    model_type = args.type

    if model_name:
        model_names = [model_name]
    else:
        with open("models.txt") as fh:
            model_names = fh.readlines()
            model_names = [model.rstrip() for model in model_names]

    batches = [batch] if batch else [1, 4]
    seqs = [seq] if seq else [32, 64, 128, 256]

    for model_name in model_names:
        for batch in batches:
            for seq in seqs:
                compile(model_name, batch, seq, target, use_trt, model_type)


if __name__ == "__main__":
    main()
