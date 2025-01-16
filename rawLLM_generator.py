from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Any, Optional, List
import numpy as np
import re
from tqdm import tqdm
import jsonlines


device = "cuda:3"
max_new_tokens = 512
model_path = "meta-llama/Llama-3.1-8B-Instruct"
num_votes = 1
input_file = "./math_testset_annotation.jsonl"

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
        # max_new_tokens=256,
        num_return_sequences=num_votes,
        do_sample=True,
        top_k=32,
        temperature=0.7, ## 可以调整一下提升表现 
        # stopping_criteria=stopping_criteria,
        # repetition_penalty=1.1,
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts[0]

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

prompt_template = (
    "You are a math problem solver. Please answer the question step by step. At the begin of each step please signify the step No. in the form 'Step No.:'. "
    r"Please write the final answer with \boxed{}\n"
    r"Please write the final answer with \boxed{}\n!!!!"
)

# First count total lines in file
total_lines = sum(1 for line in jsonlines.open(input_file))

# Read sampled line numbers from file
with open('sampled_lines.txt', 'r') as f:
    sampled_lines = [int(line.strip()) for line in f if line.strip()]

output_file = "./rawLLM_sampling_cnt"+str(len(sampled_lines))+"_result.jsonl"
print("the output_file is: "+output_file)

# Read only the sampled lines
with jsonlines.open(input_file) as reader:
    for line_num, item in tqdm(enumerate(reader)):
        if line_num not in sampled_lines:
            continue
        # print("the line_num and item is :", line_num, item)
        Question = item["question"]
        prompt = prompt_template + "Question:{}\n".format(Question)
        print("#####the final prompt is#####: "+prompt)
        i=0
        while True: # loop until it has \boxed{} format answer output
                       
            # 获取生成的文本，去掉prompt部分
            text = generate(model, tokenizer, prompt)[len(prompt) :]
            print("#####the result text is#####: ",text)
            
            if r"\boxed" not in text:
                continue
            
            if r"\boxed" in text or i == 5: # all iter times should not be greater than 5
                final_answer = extract_boxed_content(text)
                print("#####the final_answer is#####: ",final_answer)
                output_item = {"question": Question, "answer": final_answer}
                # Use write mode ('w') for first item, append ('a') for others
                mode = "w" if line_num == sampled_lines[0] else "a"
                with jsonlines.open(output_file, mode=mode) as writer:
                    writer.write(output_item)
                break
            i+=1