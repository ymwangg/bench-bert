import onnx
import numpy as np
import time
import sys
import argparse
import onnxruntime as rt

parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--backend", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--N", type=int, default=1000, required=False)

args = parser.parse_args()
model_name = args.model
model_path = "models/{}/{}.onnx".format(model_name, model_name)
backend = args.backend
batch, seq = args.batch, args.seq
N = args.N

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

options = rt.SessionOptions()
#options.add_session_config_entry('session.disable_prepacking', '1')
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
#options.enable_profiling = True
sess = rt.InferenceSession(model_path, options, providers=['CUDAExecutionProvider'])
#if backend == "mkl":
#    sess = rt.InferenceSession(model_path, options, providers=['DnnlExecutionProvider'])
#    print("using mkl")
#elif backend == "cpu":
#    sess = rt.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
#    print("using cpu")

model = onnx.load(model_path)
output_names = [out.name for out in model.graph.output]
sess.run(output_names, feed_dict)

t1 = time.time()
for _ in range(N):
    onnx_res = sess.run(output_names, feed_dict)
t2 = time.time()
dt = t2 - t1
print(dt/N*1000)
#sess.disable_profiling()
print("onnx_res sum=", np.sum(onnx_res[0]))
