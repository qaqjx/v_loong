import time
import lmcache_vllm
from lmcache_vllm.vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import random
import os
from utils.args import parse_arguments
from utils.prompt import get_generate_prompt,get_docs
from utils.util import count_lines, logger
from blend_utils import compute_rl
from lmcache_vllm.blend_adapter import (append_separator,
                                        combine_input_prompt_chunks)
from vllm import LLM as vLLM, SamplingParams as vSamplingParams

# Update the model_name to switch the remote URL
model_name = "Qwen/Qwen2-7B"

def split_text_by_fragments(A, B):
    # 存储分割点及其对应的内容
    split_points = []
    
    # 将 B 中的每个片段在 A 中查找位置，并记录起始和结束位置
    for fragment in B:
        start = A.find(fragment)
        if start != -1:  # 如果片段存在于 A 中
            end = start + len(fragment)
            split_points.append((start, end, fragment))
    
    # 按照起始位置排序（确保按顺序分割）
    split_points.sort(key=lambda x: x[0])
    
    # 初始化结果列表
    result = []
    last_end = 0
    
    # 遍历所有分割点，逐步切割 A
    for start, end, fragment in split_points:
        # 添加片段前的部分
        if last_end < start:
            result.append(A[last_end:start])  # 添加未匹配的部分
        # 添加匹配的片段
        result.append(fragment)
        last_end = end  # 更新上一个结束位置
    
    # 添加最后一个片段之后的部分
    if last_end < len(A):
        result.append(A[last_end:])
    
    return result


def precompute_kv(text_chunk, llm):
    sampling_params_prefix = SamplingParams(temperature=0.0,
                                            top_p=0.95,
                                            max_tokens=1)
    text_chunk = append_separator(text_chunk)
    llm.generate([text_chunk], sampling_params_prefix)

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    logger.debug(f"args: {args}")

    ttft_blend = []
    ttft_full = []

    f1_blend = []
    f1_full = []

    ## load data and evaluate
    llm = LLM(model=model_name, gpu_memory_utilization=0.6,quantization="fp8")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm.set_tokenizer(tokenizer)

    with open(args.input_path, "r") as file:
        line = file.readline()
        item = json.loads(line)
        # print(item["question"])
        print("item",item["doc"])
        docs = item["doc"]
        get_generate_prompt(args, item)  
        print("docs",len(docs))

        docs_str = []
        for doc in docs:
            doc_file = "/home/xujie/v_loong/data/doc/paper/" + doc
            with open(doc_file, 'r') as txt_file:
                content = txt_file.read()
                doc_name = content.split('\n', 1)[0].strip("#").strip()
                doc_str = f"{doc_name}\n" + content + "\n\n"
                docs_str.append(doc_str)
                # doc_str = append_separator(doc_str)
                precompute_kv(doc_str, llm)


        time.sleep(3)
        sampling_params_generation = SamplingParams(temperature=0.0,
                                        top_p=0.95,
                                        max_tokens=300)
        user_prompt = split_text_by_fragments(item["prompt"],docs_str)
        # print("user_prompt",len(user_prompt))
        user_prompt = combine_input_prompt_chunks(user_prompt)

        # find the difference of user_prompt and item["prompt"]
        # for i in range(len(user_prompt)):
        #     if user_prompt[i] != item["prompt"][i]:
        #         print("user_prompt[i]",user_prompt[i:i+300])
        #         print("item_prompt[i]",item["prompt"][i:i+300])
        #         break

        # print("user_prompt",user_prompt)

        output = llm.generate(user_prompt[:30000], sampling_params_generation)


        res = output[0].outputs[0].text
        print("res", res[:200])  # Output the first 200 characters of res
        answers = item["answer"]
        print("answers",answers)
        # ttft_blend.append(ttft)
        fl = max([compute_rl(res, answer) for answer in answers])
        print("fl",fl)
        f1_blend.append(fl)
    lmcache_vllm.close_lmcache_engine()

    print("blend finish")

    # if not os.path.exists(args.output_process_path) or (args.debug_num > 0 and count_lines(args.output_process_path) != args.debug_num) or (args.debug_num < 0 and count_lines(args.output_process_path) != count_lines(args.input_path)):
    #     llm = vLLM(model=model_name, gpu_memory_utilization=0.95,quantization="fp8")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     llm.set_tokenizer(tokenizer) 
        
    #     with open(args.input_path, "r") as file:
    #         line = file.readline()
    #         item = json.loads(line)
    #         get_generate_prompt(args, item)  

    #         sampling_params_generation = vSamplingParams(temperature=0.0,
    #                                         top_p=0.95,
    #                                         max_tokens=300)

    #         output = llm.generate(item["prompt"], sampling_params_generation)

    #         res = output[0].outputs[0].text
    #         # TODO(Jiayi): please move this to utils
    #         # res = res.lstrip('\n').split('\n')[0]
    #         print("full res",res)
    #         answers = item["answer"]
            
    #         fl = max([compute_rl(res, answer) for answer in answers])
    #         f1_full.append(fl)
    
    print("---------------Result Summary---------------------")
    # print(f"TTFT with cache: {np.mean(ttft_blend)}")
    # print(f"TTFT with full prefill: {np.mean(ttft_full)}")
    print(f"F1 with cache: {np.mean(f1_blend)}")
    print(f"F1 with full prefill: {np.mean(f1_full)}")

