from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Any, Optional, List
from context_cite import ContextCiter
import numpy as np
import re
import jsonlines
from tqdm import tqdm
from itertools import islice

device='cuda:0'
verifier_device = 'cuda:0'
max_new_tokens = 512
verifier_max_new_tokens = 256
model_path = "meta-llama/Llama-3.1-8B-Instruct"
verifier_model_path = "meta-llama/Llama-3.1-8B-Instruct" #gemma(verifier模型)
num_votes = 1
input_file = "./MATH_500.jsonl"
output_file = "./output_0121-default_config_no_refine_full.jsonl"
start_line = 30
end_line = 499
threshold = 1e-7

tokenizer = AutoTokenizer.from_pretrained(model_path, padding = False)
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, 
        ##low_cpu_mem_usage=True,
        #device_map="auto"
        ).to(device)


# verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_path, padding = False)
# # tokenizer.padding_side = 'right'
# # tokenizer.pad_token = tokenizer.eos_token

# verifier_model = AutoModelForCausalLM.from_pretrained(verifier_model_path, 
#         torch_dtype=torch.bfloat16, 
#         ##low_cpu_mem_usage=True,
#         #device_map="auto"
#         ).to(verifier_device )
# verifier_tokenizer = tokenizer
# verifier_model = model

stop_words = ["###"," ###", "#", "#####", "### ", "##### ", " #####"]
stop_words_ids = [tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]
    
class sub_ContextCiter(ContextCiter):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template = "Context: {context}\n\nInstruction: {query}",
        num_ablations = 64,
    ) -> None:
        super().__init__(model, tokenizer, context, query, generate_kwargs = generate_kwargs, prompt_template = prompt_template, num_ablations=num_ablations)
    
    # def _get_prompt_ids(
    #     self,
    #     mask = None,
    #     return_prompt: bool = False,
    # ):
    #     context = self.partitioner.get_context(mask)
    #     prompt = self.prompt_template.format(context=context, query=self.query)
    
    #     chat_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

    #     if return_prompt:
    #         return chat_prompt_ids, prompt
    #     else:
    #         return chat_prompt_ids
    
    def _get_prompt_ids(
        self,
        mask = None,
        return_prompt: bool = False,
    ):
        context = self.partitioner.get_context(mask)
        final_prompt = self.prompt_template.format(context=context, query=self.query)
        system = r'You are a math problem solver trying to solve the Question demonstrated in the context step by step. If the answer can be derived from previous step, please output the final answer with \boxed{} directly!If not, please output the next potential step based on the previous steps and question with Step No.in the front. . Please only output 1 step everytime.'
        few_shot_context1 = r'''Question: Evaluate $i^5+i^{-25}+i^{45}$.
        Step 1: The powers of $i$ follow a cyclical pattern: $i^1 = i$, $i^2 = -1$, $i^3 = -i$, $i^4 = 1$, and then the cycle repeats. 
        We can use this pattern to simplify each term in the expression. 
        '''
        few_shot_answer1 = r'Step 2: For $i^5$, we can rewrite it as $i^4 \cdot i$, which simplifies to $1 \cdot i = i$. For $i^{-25}$, we can rewrite it as $\frac{1}{i^{25}}$. Since $i^{25}$ is equivalent to $(i^4)^6 \cdot i$, and $i^4 = 1$, we have $i^{25} = 1^6 \cdot i = i$.'
        few_shot_context2 = r'''Question: What power of 4 is equal to 8? \
        Step 1: Express 8 as a power of 2: $8 = 2^3$
        Step 2:Express 4 as a power of 2: $4 = 2^2$
        Step 3:Equate the exponents: $2x = 3$ $x=\frac{3}{2}$'''
        few_shot_answer2 = r'''Final answer can be derived from previous steps, which is \boxed{\frac{3}{2}}
        '''
        prompt1 =  self.prompt_template.format(context=few_shot_context1, query=self.query) 
        prompt2 =  self.prompt_template.format(context=few_shot_context2, query=self.query)  
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt2}, {"role": "model", "content": few_shot_answer2}]
        # messages.extend([{"role": "user", "content": prompt2}, {"role": "assistant", "content": few_shot_answer2}])
        messages.append({"role": "user", "content": final_prompt})
        # print(messages)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        if return_prompt:
            return chat_prompt_ids, chat_prompt
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
    inputs = tokenizer(prompt, return_tensors="pt").to(verifier_device)
    print('*'*80)
    print('verification prompt:\n',prompt)
    print('*'*80)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        #num_return_sequences=num_votes,
        #do_sample=True,
        #top_k=50,
        #top_p=0.95,
        #temperature=0.3,
        #num_beams=2,
        # repetition_penalty=1.2,
        #no_repeat_ngram_size=3
        stopping_criteria=stopping_criteria,
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # print(generated_texts[0])
    return generated_texts[0] 

def extract_reasons(text: str) -> str:
    """
    提取文本中'reasons:'之后到第一个#之前的内容，并去除连续重复3次以上的句子
    Args:
        text: 包含reasons:的文本
    Returns:
        str: reasons后面到第一个#之前的内容，去除重复后的结果，如果没找到返回空字符串
    """
    # 使用正则表达式查找reasons:后到第一个#之前的所有内容
    pattern = r'reasons:(.*?)(?=#|$)'  # 匹配到第一个#或文本结束
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)  # DOTALL让.也能匹配换行符
    
    if match:
        # 提取并清理文本
        reasons = match.group(1).strip()
        
        # 按句子分割文本
        sentences = re.split(r'[.!?]+\s*', reasons)
        
        # 去除空字符串
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 去除连续重复3次以上的句子
        result = []
        i = 0
        while i < len(sentences):
            count = 1
            j = i + 1
            # 计算当前句子连续重复次数
            while j < len(sentences) and sentences[j] == sentences[i]:
                count += 1
                j += 1
            
            # 根据重复次数添加句子
            if count < 3:
                result.extend([sentences[i]] * count)
            else:
                result.append(sentences[i])
            
            i = j
            
        return '. '.join(result)
    
    return ""
def verify(verifier_pipe, prompt) -> bool:
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
    for i in range(3):
        # verifier_cc = sub_ContextCiter(model, tokenizer, prompt, '', generate_kwargs=verifier_generate_kwargs, prompt_template='{context}', num_ablations=1)
        # text = generate(model, tokenizer, prompt)[len(prompt):]
        # text = verifier_cc.response
        text = verifier_generate_text(verifier_pipe, prompt, max_new_tokens)
        print('*'*80)
        print('\n verification results:\n', text)
        print('*'*80)
        
        reasons = extract_reasons(text)
        # 如果文本不含\boxed，返回True
        if r'\boxed' not in text:
            continue
        
        # 查找第一个\boxed{}中的内容
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        
        if match:
            answer = match.group(1).strip().lower()
            # 返回True如果是yes，False如果是no
            if ('no' not in answer) and ('yes' not in answer):
                continue
        
        # 如果没有找到匹配（但有\boxed），返回True
            return 'no' not in answer, reasons
    return True, reasons

def count_steps(text: str) -> int:
    # 使用正则表达式查找所有包含"Step"的实例
    pattern = r'Step \d+'
    matches = re.findall(pattern, text)
    print('Step Count:', len(matches))
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
    # print("输入文本:", text)
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

def generate_verification_prompt(question, current_step, context):
    prompt = f"Question:\n{question}\n"
    prompt += f"Context:\n{context}\n"
    prompt += f'''Verify Step: {current_step}\n
    Is this step correct?(Yes/No).
    If the step is incorrect (No), please provide the following: **Explanation:** Explain why the original step is incorrect. And respond \\boxed{{No}} on a new line.
    If the step is correct (Yes), please respond \\boxed{{Yes}}
    '''
    return prompt

def remove_prefix(string_a: str, string_b: str) -> str:
    """
    从string_a中移除以string_b开头的前缀
    Args:
        string_a: 原始字符串
        string_b: 需要移除的前缀
    Returns:
        str: 移除前缀后的字符串
    """
    if string_a.startswith(string_b):
        return string_a[len(string_b):]
    return string_a


stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])

generate_kwargs = {
    'max_new_tokens': max_new_tokens,
    #'num_return_sequences': num_votes,
    #'do_sample': True,
    #'top_k': 32,
    # 'temperature': 0.3,
    #'repetition_penalty':1.2,
    #'stopping_criteria': stopping_criteria,
}

verifier_generate_kwargs = {
    'max_new_tokens': max_new_tokens,
    #'num_return_sequences': num_votes,
    #'do_sample': True,
    #'top_k': 32,
    'temperature': 0.3,
    #'repetition_penalty':1.2,
    'stopping_criteria': stopping_criteria,
}

# verifier_pipe = pipeline(
#     "text-generation",
#     model=verifier_model_path,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device=verifier_device,  
# )

def verifier_generate_text(verifier_pipe, prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": '''
        You are a math question verifier.
        Please answer '\\boxed{yes}' or '\\boxed{no}' and the reasons to verify whether the to be verified step can be derived from the Context without hallucination or error.
        If the 'to be verified step' contains duplicate content, Please answer \\boxed{no}
        If it's not a fatal error, please do not answer \\boxed{no}. Answer \\boxed{yes} as much as possible, since we could accept minor mistake.
        Your response should be in the form of: results:\\boxed{no/yes} \n reasons:'''},
        {"role": "user", "content": '''
         Context: Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
        Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
        To be verified step: 
        factor $x^2+x-6$, which is $(x-3)(x+2)$
        '''},
        {"role": "model", "content":'''
         results:\\boxed{no}
        \\reasons: $x^2+x-6$ doesn't equal to $(x-3)(x+2)$ but $(x-2)(x+3)$.
         '''
        },
        {"role": "user", "content": '''
        Context: Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
        Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
        Step 4: the asymptotes for $x^2+x-6$ should be x=2 and x = -3.
        To be verified step: 
        So the number of asymptotes should be 2.
        '''},
        {"role": "model", "content":'''
         results:\\boxed{yes}
        \\reasons: since the asymptotes for $x^2+x-6$ is x=2 and x=-3, the number of asymptotes should be 2.
         '''
        },
        {"role": "user", "content": '''
        Context: Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
        Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
        Step 4: the asymptotes for $x^2+x-6$ should be x=2 and x = -3.
        To be verified step: 
        So the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.the number of asymptotes should be 2.
        '''},
        {"role": "model", "content":'''
         results:\\boxed{No}
        \\reasons: The to be verifier step contains duplicate content. Please remove duplicate content.
         '''
        },
        {"role": "user", "content": prompt},
    ]
    outputs = verifier_pipe(messages, do_sample=True, top_p=0.95, temperature=0.7, max_new_tokens=verifier_max_new_tokens)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

# prompt_template = (
#     "You are a math problem solver. Please answer the question step by step. At the begin of each step please signify the step No. in the form 'Step No.:'. "
#     "At the end of each step please output '###' to signify the end of the step.For example, in the first step, you should write in the form 'Step 1: ...\n ###'for the first step\n\n"
#     r"Please write the final answer with \boxed{} ###\n"
# )
# query = 'You are a math problem solver. You are suppose to output the next potential step. Do not output more than 1 steps. You have to output step No.before the step. If the answer can be derived from previous steps? if yes, output \\boxed{final answer}. If not, What is the potential next step based on the previous steps and question?'


query = r'What is the potential next step or answer?'
# verifier_prompt_template = (
#     "You are a math question verifier."
#     "Question:{Question}\n Context:{Context} \n to be verified step:{verified_step}\n"
#     "The to be verified step is only one step for the solution process, so you don't need to consider whether the step solves the question or not.\n"
# )
verifier_prompt_template = '''
    You are a math question verifier.
    Please answer '\\boxed{yes}' or '\\boxed{no}' and the reasons to verify whether the to be verified step can be derived from the Context without hallucination or error.\n
    Your response should be in the form of: results:\\boxed{no/yes} \n reasons:
    
    #####
    
    Context: Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
    Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
    To be verified step: 
    factor $x^2+x-6$, which is $(x-3)(x+2)$
    
    results:\\boxed{no}
    \\reasons: $x^2+x-6$ doesn't equal to $(x-3)(x+2)$ but $(x-2)(x+3)$.
    
    #####
    
    Context: Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
    Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
    Step 4: the asymptotes for $x^2+x-6$ should be x=2 and x = -3.
    To be verified step: 
    So the number of asymptotes should be 2.
    
    results:\\boxed{yes}
    \\reasons: since the asymptotes for $x^2+x-6$ is x=2 and x=-3, the number of asymptotes should be 2.
    
    #####    
'''
verifier_prompt_template2 =  "Context:{Context} \nTo be verified step:{verified_step}\n"

self_refine_template = '''
You are a math problem solver.Since the last step is incorrect, now you have to regenerate the last step based on the previous step, question and instruction.
The instruction explains why the last step is incorrect. 

#####

Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
Step 1: to determine the asymptotes, we should find the zero point of $y=\\frac{2}{x^2+x-6}$.
Step 2: to find the zero point, we have to factor the $x^2+x-6$.
To be regenerated step: Step 3:factor $x^2+x-6$, which is $(x-3)(x+2)$
Instruction: $x^2+x-6$ doesn't equal to $(x-3)(x+2)$ but $(x-2)(x+3)$

regenerated step: steps 3:factor $x^2+x-6$, which is $(x+3)(x-2)$

#####
'''
self_refine_template2 = '{steps}\nTo be regenerated step: {target_step}\nInstruction: {instruction}\n\nregenerated step: '


regenerate_prompt_template = (
    "Please regenerate the last step based on the instruction:"
)

with jsonlines.open(input_file) as reader:
    for item in tqdm(islice(reader, start_line, end_line)):
        Question = item['question']
        # prompt = prompt_template+"Question:{}\n".format(Question)
        prompt = "Question:{}\n".format(Question)
        prompt_len = len(prompt)
        i=0
        regenerate = 0
        refine = 0
        while True:
            prompt = prompt.replace("<|eot_id|>", "\n")+"<|eot_id|>"
            print('*'*80)
            print('\nprompt:\n', prompt, '\n')
            print('*'*80)
            cc = sub_ContextCiter(model, tokenizer, prompt, query, generate_kwargs=generate_kwargs, num_ablations=1)
            try:
                generated_texts = cc.response
            except:
                print("none type")
            if count_steps(generated_texts)>=3 or len(generated_texts)<=10:
                print('regenerating this step.')
                regenerate +=1
                if regenerate<=2:
                    continue
                
            regenerate = 0
            raw_results = cc.get_attributions()
            indices = np.where(raw_results > threshold)[0]
            extract_context = [cc.sources[int(i)] for i in indices]
            filtered_context = [context for context in extract_context if context not in prompt]
            Context = '\n'.join(filtered_context)
            if refine==0:
                Context0=Context
            # verify_prompt = verifier_prompt_template + verifier_prompt_template2.format(Question = Question, Context = Context0, verified_step = generated_texts)
            verify_prompt = verifier_prompt_template2.format(Question = Question, Context = Context0, verified_step = generated_texts)
            # results, reasons = verify(verifier_pipe, verify_prompt)
            results = True
            reasons = ''
            
            if results == False and refine<=2:
                if refine==0:
                    prompt0 = prompt
                    step_prompt = prompt
                    print('*'*80)
                    print('step_prompt:',step_prompt)
                    print('*'*80)
                    print('self-refine-generated_text:',generated_texts)
                    print('*'*80)
                    prompt = self_refine_template + self_refine_template2.format(steps=step_prompt, target_step = generated_texts, instruction = reasons)
                    refine+=1
                    print('\nself-refining\n')
                    continue
                else:
                    print('*'*80)
                    print('step_prompt:',step_prompt)
                    print('*'*80)
                    print('self-refine-generated_text:',generated_texts)
                    print('*'*80)
                    prompt = self_refine_template + self_refine_template2.format(steps=step_prompt, target_step = generated_texts, instruction = reasons)
                    refine+=1
                    print('\nself-refining\n')
                    continue 
            elif refine>0:
                print('*'*80)
                print('step_prompt:',step_prompt)
                print('*'*80)
                print('self-refine-generated_text:',generated_texts)
                print('*'*80)
                prompt = prompt0
                refine = 0
            
            print('\nVerification bool results:', results)
            # print('\ngenerated steps:\n', generated_texts, end = '\n')
            if refine==0:
                prompt = prompt+generated_texts
            if r'\boxed' in generated_texts or i==25:
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



