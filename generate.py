from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

device='cuda:0'
model_path = 'meta-llama/Llama-3.1-8B-Instruct'
max_length = 1024
num_votes = 1

tokenizer = AutoTokenizer.from_pretrained(model_path, padding = False)
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
stop_words = ["###"]
stop_words_ids = [tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]

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

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])

prompt_template = (
    "You are a math problem solver. Please answer the question step by step. At the begin of each step please signify the step No. in the form 'Step No.:'. "
    "At the end of each step please output '###' to signify the end of the step.For example, in the first step, you should write in the form 'Step 1: ...\n ###' at the begining of the step\n\n"
    r"Please write the final answer with \boxed{}\n###\n"
)
prompt = prompt_template+"Question:{}\n".format("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n")
    
while True:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_votes,
        do_sample=True,
        top_k=32,
        temperature=0.7,
        stopping_criteria=stopping_criteria,
        #repetition_penalty=1.1,      
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print(generated_texts)
    prompt = generated_texts
    if r'\\boxed' in prompt[0]:
        break
