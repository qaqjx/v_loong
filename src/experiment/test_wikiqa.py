from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("inputs/wikimqa_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
query_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: "

ttft_full = []
f1_full = []
#max_ctx_len = 4096-196

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_end = [733, 28748, 16289, 28793]
    s_end_len = len(s_end)
    old_kvs = []

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len([q_ids+s_end])
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
    input_prompt = tokenizer.decode(input_ids)
    
 
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer[0], tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")