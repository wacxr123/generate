from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Any, Optional, List
import numpy as np
import re
from tqdm import tqdm
import jsonlines
import argparse

# Default constants
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device", type=str, default="auto", help='GPU device number (0,1,2,3) or "auto" for automatic device mapping'
)
parser.add_argument("--file", type=str, required=True, help="Input jsonl file path")
parser.add_argument(
    "--lineRange", type=int, nargs=2, required=True, help="Line range to process (e.g., --lineRange 1 500)"
)
parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model to use")

args = parser.parse_args()

model_path = args.model_path

# Handle device setting
device = "auto" if args.device.lower() == "auto" else f"cuda:{args.device}"

# Create pipeline directly instead of loading model and tokenizer separately
pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)


def extract_boxed_content(text: str) -> str:
    """
    提取\boxed{}中的内容，支持嵌套大括号
    Args:
        text: 包含\boxed{}的文本
    Returns:
        str: \boxed{}中的内容，如果没找到返回空字符串
    """
    # 添加调试打印
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
    r"Please write the final answer with \boxed{}. Remember, write the final answer in the '\boxed{}' annotation! \n"
)

# Replace the ranges and part logic with lineRange
start, end = args.lineRange
sampled_lines = range(start - 1, end)  # -1 for 0-based indexing
myrange = f"index[{start}-{end}]"
path = model_path.split("/")[-1]
output_file = "./rawLLM_" + myrange + path + "_result.jsonl"
print("The output_file is: " + output_file)

# Update input file to use args.file
input_file = args.file

# Read only the sampled lines
for line_num in tqdm(sampled_lines, desc="Processing sampled lines"):
    with jsonlines.open(input_file) as reader:
        for current_line_num, item in enumerate(reader):
            if current_line_num == line_num:
                break
        print("the line_num is :", line_num)
        Question = item["question"]
        prompt = prompt_template + "Question:{}\n".format(Question)
        print("#####the final prompt is#####: " + prompt)
        i = 0
        while True:  # loop until it has \boxed{} format answer output
            text = pipe(prompt)
            print("#####the pipeline result text is#####: ", text)

            if r"\boxed" not in text:
                continue

            if r"\boxed" in text or i == 5:  # all iter times should not be greater than 5
                final_answer = extract_boxed_content(text)
                print("#####the final_answer is#####: ", final_answer)
                output_item = {"question": Question, "answer": final_answer}
                # Use write mode ('w') for first item, append ('a') for others
                mode = "w" if line_num == sampled_lines[0] else "a"
                with jsonlines.open(output_file, mode=mode) as writer:
                    writer.write(output_item)
                break
            print("No \boxed found, regenerating......\n")
            i += 1
