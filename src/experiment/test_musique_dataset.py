import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("/home/xujie/v_loong/src/experiment/inputs/musique_s.json")

lengths = []
context_lens = []
for ex in eval_dataset:
    answers = ex["answers"]
    ctxs = ex["ctxs"]
    context_len = 0
    for ctx in ctxs:
        context_len += len(ctx["text"])
        lengths.append(len(ctx["text"]))
    context_lens.append(context_len)
    # concatenated_ctxs = " ".join(ctx["text"] for ctx in ctxs)        

print("---------------Result Summary---------------------")
print(f"Max context length: {np.max(context_lens)}")
print(f"Min context length: {np.min(context_lens)}")
print(f"Mean context length: {np.mean(context_lens)}")

print(f"Max length: {np.max(lengths)}")
print(f"Min length: {np.min(lengths)}")
print(f"Mean length: {np.mean(lengths)}")