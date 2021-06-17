import sys
import tvm
from tvm import relay
import onnx
import argparse


parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--target", type=str, default='llvm -mcpu=skylake-avx512 -libs=mkl,mlas')

args = parser.parse_args()
model_path = args.model
batch, seq = args.batch, args.seq
target = args.target
print("target = {}".format(target))

prefix = model_path[:-5]
print(prefix)
model = onnx.load(model_path)

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

def save_model(lib, prefix):
    prefix = "{}-{}-{}".format(prefix, batch, seq)
    lib.export_library("{}.so".format(prefix))
    with open("{}.json".format(prefix), 'w') as fh:
        fh.write(lib.graph_json)
    with open("{}.params".format(prefix), 'wb') as fh:
        fh.write(relay.save_param_dict(lib.params))

mod, par = relay.frontend.from_onnx(model, shape=shape, freeze_params=True)
with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    #executable = relay.vm.compile(mod, params = params, target=target)
    lib = relay.build(mod, params=par, target=target)

save_model(lib, prefix)
