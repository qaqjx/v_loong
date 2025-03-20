import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_fewshot_prompt, compute_rl
from pathlib import Path
from itertools import chain
from lmcache_vllm.blend_adapter import (OfflineKVPreCompute,append_separator,
                                        combine_input_prompt_chunks)
from lmcache_vllm.vllm import LLM, SamplingParams
import torch
import time


eval_dataset = load_dataset("inputs/samsum.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)
offline_precompute = OfflineKVPreCompute(llm)

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
    doc_chunk_ids = [(doc)[1:] for doc in doc_prompts]
    q_ids = (q_prompt)[1:]
    
    for prompt in doc_chunk_ids:
        offline_precompute.precompute_kv(prompt)

    time.sleep(3)

    # Create an tokenizer and LLM.
    intput_prompt = [prefix_prompt] + doc_chunk_ids + [q_ids]
    user_promt = combine_input_prompt_chunks(intput_prompt)    
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    
    output = llm.generate(user_promt, sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    ttft_full.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full.append(rl)
    print("------------")
    

print("---------------Result Summary---------------------")
print(f"TTFT with cacheblend prefill: {np.mean(ttft_full)}")
print(f"rl with cacheblend prefill: {np.mean(rl_full)}")