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
    from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_expr, rewrite, is_op
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

    if use_trt:
        print("using trt")
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
