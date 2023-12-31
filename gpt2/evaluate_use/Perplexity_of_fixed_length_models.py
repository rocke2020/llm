# https://huggingface.co/docs/transformers/perplexity

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "/mnt/nas1/huggingface/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)
# torch.Size([1, 287644])
ic(encodings.input_ids.shape)
# max_length: 1024, seq_len: 287644
ic(max_length, seq_len)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break
ic(len(nlls))  # 561
# ppl.item(): 25.199005126953125
ppl = torch.exp(torch.stack(nlls).mean())
ic(ppl.item())
