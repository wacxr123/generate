from transformers import AutoTokenizer, AutoModelForCausalLM

device='cuda:0'
model_path = 'meta-llama/Llama-3.1-8B-Instruct'
max_length = 1024
num_votes = 1

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
stop_words_ids = [
tokenizer.encode(stop_word, add_prefix_space = False) for stop_word in ["###","#"]]

prompt_template = (
    "You are a math problem solver. Please answer the question step by step. At the begin of each step please signify the step No. in the form 'Step No.:'. For example, in the first step, you should write 'Step 1:' at the begining of the step\n"
    "At the end of each step please output '###' to signify the end of the step.\n"
    r"Please write the answer with \boxed"
    "Question:{question}\n" 
)
prompt = prompt_template.format(
    question="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_length=max_length,
    num_return_sequences=num_votes,
    do_sample=True,
    top_k=32,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
    #repetition_penalty=1.1,      
)

generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print(generated_texts)
