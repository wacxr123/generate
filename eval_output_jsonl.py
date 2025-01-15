import json
import sys

def calculate_accuracy(standard_file, model_file):
    with open(standard_file, 'r', encoding='utf-8') as f:
        standard_data = {item["question"]: item["answer"] for line in f for item in [json.loads(line)]}
    
    with open(model_file, 'r', encoding='utf-8') as f:
        model_data = {item["question"]: item["answer"] for line in f for item in [json.loads(line)]}
    
    correct = 0
    total_questions = 0
    for question, answer in model_data.items():
        if question in standard_data and standard_data[question] == answer:
            correct += 1
        total_questions += 1
    
    return correct / total_questions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_output_jsonl.py <standard_file> <model_file>")
        sys.exit(1)
    standard_file = sys.argv[1]
    model_file = sys.argv[2]
    accuracy = calculate_accuracy(standard_file, model_file)
    print("Accuracy:", accuracy)
