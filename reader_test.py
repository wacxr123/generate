import jsonlines
from tqdm import tqdm


sampled_lines = range(0, 1250) 
myrange = "index[1-1250]" 
output_file = "./rawLLM_" + myrange + "_result.jsonl"
print("The output_file is: " + output_file)

# Read only the sampled lines
input_file = "./math_testset_annotation.jsonl"

for line_num in tqdm(sampled_lines, desc="Processing partial lines"):
    with jsonlines.open(input_file) as reader:
        for current_line_num, item in enumerate(reader):
            if current_line_num == line_num:
                break
        print("the line_num is :", line_num,item)