from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_fewshot_prompt, compute_rl
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("inputs/samsum.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"

ttft_full = []
rl_full = []

max_ctx_len = 3400
#TODO (Jiayi): fix filler tokens at the begining or pass in tokenizer
for sample_idx, ex in enumerate(eval_dataset):
    answers = ex["answers"]
    doc_prompts, q_prompt = build_fewshot_prompt(ex)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    
    # drop last few-shot examples if exceeding max_ctx_len
    while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
        del_idx = int(len(doc_chunk_ids)/2)
        del doc_chunk_ids[del_idx]
    
    # skip if all ctxs are dropped
    if len(doc_chunk_ids)==0:
        continue
                
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    s_start_full = tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_end = []
    s_end_len = len(s_end)

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)
    
    
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    ttft_full.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full.append(rl)
    print("------------")
    

print("---------------Result Summary---------------------")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"rl with full prefill: {np.mean(rl_full)}")