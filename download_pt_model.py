import argparse
import torch
from transformers import BertModel
import numpy as np


parser = argparse.ArgumentParser(description="Process input args")
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()
model_name = args.model
print("Download {}.pt".format(model_name))
model = BertModel.from_pretrained(model_name, torchscript=True)

batch, seq = 1, 128
shape = {
    "input_ids" : (batch, seq),
    "attention_mask" : (batch, seq),
}

feed_dict = {
    'input_ids' : torch.tensor(np.random.randint(0, 10000, size=[batch,seq]).astype("int64")),
    'attention_mask' : torch.tensor(np.zeros([batch,seq]).astype("int64")),
}

if "distilbert" not in model_name and "roberta" not in model_name:
    shape["token_type_ids"] = (batch, seq)
    feed_dict["token_type_ids"] = torch.tensor(np.zeros([batch,seq]).astype("int64"))
    traced_model = torch.jit.trace(model, [feed_dict['input_ids'], feed_dict['attention_mask'], feed_dict['token_type_ids']])
else:
    traced_model = torch.jit.trace(model, [feed_dict['input_ids'], feed_dict['attention_mask']])

torch.jit.save(traced_model, "pt_models/{}/{}.pt".format(model_name, model_name))