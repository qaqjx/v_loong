from lmcache_vllm.blend_adapter import (OfflineKVPreCompute,append_separator,
                                        combine_input_prompt_chunks)
from lmcache_vllm.vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
import time


eval_dataset = load_dataset("inputs/musique_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
offline_precompute = OfflineKVPreCompute(llm)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [(doc)[1:] for doc in doc_prompts]
    q_ids = (q_prompt)[1:]

    for prompt in doc_chunk_ids:
        offline_precompute.precompute_kv(prompt)

    time.sleep(3)

    doc_chunk_ids = [prefix_prompt] + doc_chunk_ids + [q_ids]
    user_promt = combine_input_prompt_chunks(doc_chunk_ids)
    user_promt = "[INST]" + user_promt + "[/INST]"
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    output = llm.generate(user_promt, sampling_params)
    res = output[0].outputs[0].text
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"TTFT with cacheblend prefill: {np.mean(ttft_full)}")
print(f"F1 with cacheblend prefill: {np.mean(f1_full)}")