from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Any, Optional, List
from context_cite import ContextCiter
import numpy as np
import re
import jsonlines
from tqdm import tnrange
from itertools import islice

device='cuda:0'
max_new_tokens = 512
verifier_max_new_tokens = 256
model_path = "meta-llama/Llama-3.1-8B-Instruct"
num_votes = 1
input_file = "./math_testset_annotation.jsonl"
output_file = "./output1.jsonl"
start_line = 0
end_line = 5

tokenizer = AutoTokenizer.from_pretrained(model_path, padding = False)
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
stop_words = ["###"," ###", "#"]
stop_words_ids = [tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]

class sub_ContextCiter(ContextCiter):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template = '',
    ) -> None:
        super().__init__(model, tokenizer, context, query, generate_kwargs = generate_kwargs, prompt_template = prompt_template)

    def _get_prompt_ids(
        self,
        mask = None,
        return_prompt: bool = False,
    ):
        context = self.partitioner.get_context(mask)
        prompt = self.prompt_template.format(context=context, query=self.query)
    
        chat_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if return_prompt:
            return chat_prompt_ids, prompt
        else:
            return chat_prompt_ids
    
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        self.stops = stops if stops is not None else []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop_ids in self.stops:
            stop_ids_tensor = torch.tensor(stop_ids).to(input_ids.device)
            for seq_idx in range(input_ids.shape[0]):
                if input_ids[seq_idx].size(0) >= len(stop_ids):
                    if torch.all(input_ids[seq_idx][-len(stop_ids):] == stop_ids_tensor):
                        return True
        return False
 
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_return_sequences=num_votes,
        do_sample=True,
        top_k=32,
        temperature=0.7,
        stopping_criteria=stopping_criteria,
        #repetition_penalty=1.1,      
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
    pattern = r'reasons:(.*?)(?=\\boxed|$)'  # 匹配到下一个\boxed或文本结束
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)  # DOTALL让.也能匹配换行符
    
    if match:
        # 提取并清理文本
        reasons = match.group(1).strip()
        return reasons
    
    return ""
    
def verify(model, tokenizer, prompt) -> bool:
    """
    生成文本并检查第一个\boxed{}中的答案
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示
    Returns:
        bool: True 如果是 yes 或没有 \boxed，False 如果是 no
    """
    # 获取生成的文本，去掉prompt部分
    text = generate(model, tokenizer, prompt)[len(prompt):]
    print('\n verification results:\n', text)
    
    reasons = extract_reasons(text)
    # 如果文本不含\boxed，返回True
    if r'\boxed' not in text:
        return True, reasons
    
    # 查找第一个\boxed{}中的内容
    pattern = r'\\boxed{([^}]*)}'
    match = re.search(pattern, text)
    
    if match:
        answer = match.group(1).strip().lower()
        # 返回True如果是yes，False如果是no
        return answer == 'yes', reasons
    
    # 如果没有找到匹配（但有\boxed），返回True
    return True, reasons

def count_steps(text: str) -> int:
    # 使用正则表达式查找所有包含"Step"的实例
    pattern = r'Step \d+'
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
    if r'\boxed' not in text:
        return ""
    # 找到\boxed{后的位置
    start_pos = text.find(r'\boxed{')
    start_idx = start_pos + len(r'\boxed{')  # 使用len()更清晰
    # 计数左右大括号来处理嵌套情况
    count = 1
    current_idx = start_idx
    while count > 0 and current_idx < len(text):
        if text[current_idx] == '{':
            count += 1
        elif text[current_idx] == '}':
            count -= 1
        current_idx += 1
    if count == 0:
        result = text[start_idx:current_idx - 1].strip()
        return result
    return ""


stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])

generate_kwargs = {
    'max_new_tokens': max_new_tokens,
    'num_return_sequences': num_votes,
    'do_sample': True,
    'top_k': 32,
    'temperature': 0.7,
    'stopping_criteria': stopping_criteria,
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
verifier_prompt_template2 =  r"Your response should be in the form of: results:\boxed{yes} (or \boxed{no})\n reasons:"

regenerate_prompt_template = (
    "Please regenerate the last step based on the instruction:"
)

with jsonlines.open(input_file) as reader:
    for item in tnrange(islice(reader, start_line, end_line)):
        Question = item['question']
        prompt = prompt_template+"Question:{}\n".format(Question)
        prompt_len = len(prompt)
        i=0
        regenerate = 0
        refine = 0
        while True:
            cc = sub_ContextCiter(model, tokenizer, prompt, '', generate_kwargs=generate_kwargs, prompt_template='{context}')
            generated_texts = cc.response
            if count_steps(generated_texts)>=3:
                print('regenerating this step.')
                regenerate +=1
                if regenerate<=2:
                    continue
                
            regenerate = 0
            raw_results = cc.get_attributions()
            indices = np.where(raw_results > 1e-7)[0]
            extract_context = [cc.sources[int(i)] for i in indices]
            filtered_context = [context for context in extract_context if context not in prompt_template]
            Context = '\n'.join(filtered_context)
            verify_prompt = verifier_prompt_template.format(Question = Question, Context = Context, verified_step = generated_texts)+verifier_prompt_template2
            results, reasons = verify(model, tokenizer, verify_prompt)
            if results == False and refine<=2:
                if refine==0:
                    prompt0 = prompt
                    prompt = prompt+regenerate_prompt_template+reasons
                    refine+=1
                    print('\nself-refining\n')
                    continue
                else:
                    prompt = prompt0+regenerate_prompt_template+reasons
                    refine+=1
                    print('\nself-refining\n')
                    continue 
            elif refine>0:
                prompt = prompt0
                refine = 0
            
            print('\nVerification bool results:', results)
            print('\ngenerated steps:\n', generated_texts, end = '\n')
            prompt = prompt+generated_texts
            if r'\boxed' in generated_texts or i==30:
                final_answer = extract_boxed_content(generated_texts)
                output_item = {
                'question': Question,
                'answer': final_answer
            }
        
                # 以追加模式写入结果
                with jsonlines.open(output_file, mode='a') as writer:
                    writer.write(output_item)
                break
            i+=1



