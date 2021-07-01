#!/usr/bin/env python
import numpy as np
import time
import sys
import argparse
import torch


def benchmark(model_path, batch, seq, backend, N=1):
    shape = {
        "input_ids": (batch, seq),
        "attention_mask": (batch, seq),
    }

    if backend == "cpu":
        feed_dict = [
            torch.tensor(
                np.random.randint(0, 10000, size=[batch, seq]).astype("int64")
            ),
            torch.tensor(np.ones([batch, seq]).astype("int64")),
        ]
        if "distilbert" not in model_path and "roberta" not in model_path:
            shape["token_type_ids"] = (batch, seq)
            feed_dict.append(torch.tensor(np.zeros([batch, seq]).astype("int64")))
    else:
        feed_dict = [
            torch.tensor(
                np.random.randint(0, 10000, size=[batch, seq]).astype("int64")
            ).cuda(),
            torch.tensor(np.ones([batch, seq]).astype("int64")).cuda(),
        ]
        if "distilbert" not in model_path and "roberta" not in model_path:
            shape["token_type_ids"] = (batch, seq)
            feed_dict.append(
                torch.tensor(np.zeros([batch, seq]).astype("int64")).cuda()
            )

    loaded_model = torch.jit.load(model_path)
    if backend == "gpu":
        loaded_model.to("cuda")
    loaded_model.eval()
    for p in loaded_model.parameters():
        p.requires_grad_(False)

    for _ in range(10):
        res = loaded_model(*feed_dict)
    torch.cuda.synchronize()

    t1 = time.time()
    for _ in range(N):
        res = loaded_model(*feed_dict)
    torch.cuda.synchronize()
    t2 = time.time()

    dt = t2 - t1
    inf_time = dt / N * 1000
    return inf_time


def main():
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument(
        "--backend", type=str, required=False, default="cpu", choices=["cpu", "gpu"]
    )
    parser.add_argument("--batch", type=int, required=False)
    parser.add_argument("--seq", type=int, required=False)
    parser.add_argument("--N", type=int, required=False, default=100)
    args = parser.parse_args()
    model_name = args.model
    batch, seq = args.batch, args.seq
    backend = args.backend
    N = args.N

    if model_name:
        model_names = [model_name]
    else:
        with open("models.txt") as fh:
            model_names = fh.readlines()
            model_names = [model.rstrip() for model in model_names]

    batches = [batch] if batch else [1, 4]
    seqs = [seq] if seq else [32, 64, 128, 256]

    for batch in batches:
        print(
            "---------------begin profiling PT batch={}------------------".format(batch)
        )
        for model_name in model_names:
            model_path = "pt_models/{}/{}.pt".format(model_name, model_name)
            line = "{}".format(model_name, batch)
            for seq in seqs:
                latency = benchmark(model_path, batch, seq, backend, N=N)
                line += ",{}".format(latency)
            print(line)


if __name__ == "__main__":
    main()
