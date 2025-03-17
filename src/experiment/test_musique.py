from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("/home/xujie/v_loong/src/experiment/inputs/musique_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
query_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: "

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []
#max_ctx_len = 4096-196

for ex in eval_dataset:
    answers = ex["answers"]
    ctxs = ex["ctxs"]

    concatenated_ctxs = " ".join(ctx["text"] for ctx in ctxs)
    input_text = prefix_prompt + concatenated_ctxs + query_prompt + ex["question"]
        
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=300)
    
    output = llm.generate(input_text, sampling_params)
    res = output[0].outputs[0].text
    # print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    # print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer[0], tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
# print(f"TTFT with cache: {np.mean(ttft_blend)}")
# print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with cache: {np.mean(f1_blend)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")