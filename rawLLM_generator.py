from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Any, Optional, List
from context_cite import ContextCiter
import numpy as np
import re
import jsonlines
from tqdm import tqdm
from itertools import islice
import random

device = "cuda:3"
max_new_tokens = 512
verifier_max_new_tokens = 256
model_path = "meta-llama/Llama-3.1-8B-Instruct"
num_votes = 1
input_file = "./math_testset_annotation.jsonl"
output_file = "./output_z_cxr.jsonl"


tokenizer = AutoTokenizer.from_pretrained(model_path, padding=False)
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # why half presision here
    ##low_cpu_mem_usage=True,
    # device_map="auto"
).to(device)
stop_words = ["###", " ###", "#"]
stop_words_ids = [tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]



def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        num_return_sequences=num_votes,
        do_sample=True,
        top_k=32,
        temperature=0.7,
        stopping_criteria=stopping_criteria,
        # repetition_penalty=1.1,
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # print(generated_texts[0])
    return generated_texts[0]


def extract_reasons(text: str) -> str:
    """
    提取文本中'reasons:'之后的内容
    Args:
        text: 包含reasons:的文本
    Returns:
        str: reasons后面的内容，如果没找到返回空字符串
    """
    # 使用正则表达式查找reasons:后的所有内容
    pattern = r"reasons:(.*?)(?=\\boxed|$)"  # 匹配到下一个\boxed或文本结束
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)  # DOTALL让.也能匹配换行符

    if match:
        # 提取并清理文本
        reasons = match.group(1).strip()
        return reasons

    return ""


def count_steps(text: str) -> int:
    # 使用正则表达式查找所有包含"Step"的实例
    pattern = r"Step \d+"
    matches = re.findall(pattern, text)
    return len(matches)


def extract_boxed_content(text: str) -> str:
    """
    提取\boxed{}中的内容，支持嵌套大括号
    Args:
        text: 包含\boxed{}的文本
    Returns:
        str: \boxed{}中的内容，如果没找到返回空字符串
    """
    # 添加调试打印
    print("输入文本:", text)
    if r"\boxed" not in text:
        return ""
    # 找到\boxed{后的位置
    start_pos = text.find(r"\boxed{")
    start_idx = start_pos + len(r"\boxed{")  # 使用len()更清晰
    # 计数左右大括号来处理嵌套情况
    count = 1
    current_idx = start_idx
    while count > 0 and current_idx < len(text):
        if text[current_idx] == "{":
            count += 1
        elif text[current_idx] == "}":
            count -= 1
        current_idx += 1
    if count == 0:
        result = text[start_idx : current_idx - 1].strip()
        return result
    return ""



generate_kwargs = {
    "max_new_tokens": max_new_tokens,
    "num_return_sequences": num_votes,
    "do_sample": True,
    "top_k": 32,
    "temperature": 0.7,
}

prompt_template = (
    "You are a math problem solver. Please answer the question step by step. At the begin of each step please signify the step No. in the form 'Step No.:'. "
    "At the end of each step please output '###' to signify the end of the step.For example, in the first step, you should write in the form 'Step 1: ...\n ###'for the first step\n\n"
    r"Please write the final answer with \boxed{} ###\n"
)

verifier_prompt_template = (
    "You are a math question verifier. you will be provided a step in solving a mathematical problem that needs validation. I will also give you the context from which this step is derived and specify what the mathematical problem is. "
    "The to be verified step is only one step for the solution process, so you don't need to consider whether the step solves the question or not.\n"
    "Question:{Question}\n Context:{Context} \n to be verified step:{verified_step}\n"
    "Please answer 'yes' or 'no' and the reasons to verify whether the to be verified step can be derived from the Question and Context without hallucination or error.\n"
)
verifier_prompt_template2 = r"Your response should be in the form of: results:\boxed{yes} (or \boxed{no})\n reasons:"

regenerate_prompt_template = "Please regenerate the last step based on the instruction:"

# First count total lines in file
total_lines = sum(1 for line in jsonlines.open(input_file))

# Randomly choose line numbers (e.g. 50 samples)
num_samples = 50 
sampled_lines = sorted(random.sample(range(total_lines), num_samples))

# Save sampled line numbers
with open('sampled_lines.txt', 'w') as f:
    f.write('\n'.join(map(str, sampled_lines)))

# Read only the sampled lines
with jsonlines.open(input_file) as reader:
    for line_num, item in tqdm(enumerate(reader)):
        if line_num not in sampled_lines:
            continue
        Question = item["question"]
        prompt = prompt_template + "Question:{}\n".format(Question)
        prompt_len = len(prompt)
        i = 0
        while True:
            indices = np.where(raw_results > 1e-7)[0]
            extract_context = [cc.sources[int(i)] for i in indices]
            filtered_context = [context for context in extract_context if context not in prompt_template]
            Context = "\n".join(filtered_context)
            verify_prompt = (
                verifier_prompt_template.format(Question=Question, Context=Context, verified_step=generated_texts)
                + verifier_prompt_template2
            )
            results, reasons = verify(model, tokenizer, verify_prompt)
            if results == False and refine <= 2:
                if refine == 0:
                    prompt0 = prompt
                    prompt = prompt + regenerate_prompt_template + reasons

            elif refine > 0:
                prompt = prompt0
                refine = 0

            print("\nVerification bool results:", results)
            print("\ngenerated steps:\n", generated_texts, end="\n")
            prompt = prompt + generated_texts
            if r"\boxed" in generated_texts or i == 30:
                final_answer = extract_boxed_content(generated_texts)
                output_item = {"question": Question, "answer": final_answer}
                # Use write mode ('w') for first item, append ('a') for others
                mode = "w" if line_num == sampled_lines[0] else "a"
                with jsonlines.open(output_file, mode=mode) as writer:
                    writer.write(output_item)
                break
            i += 1
