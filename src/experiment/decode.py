from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain



tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

## musique and wikiqa

musique_start = [733, 16289, 28793]
musique_end = [733, 28748, 16289, 28793]

print("musique start :" , tokenizer.decode(musique_start))
print("musique end :" , tokenizer.decode(musique_end))


