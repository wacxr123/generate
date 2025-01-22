import json
import sys
import sympy as sp
from sympy.parsing.latex import parse_latex

def parse_latex_e(latex_str):
    if not latex_str.strip():  # Handle empty or whitespace-only strings
        return None
    
    
    try:
        expr = parse_latex(latex_str)  # Use sympy to parse LaTeX
        return expr
    except Exception as e:
        print(f"Error parsing LaTeX: {e} for input: '{latex_str}'")
        return None

def compare_latex_answers(latex1, latex2):
    expr1 = parse_latex_e(latex1)
    expr2 = parse_latex_e(latex2)

    if expr1 is None or expr2 is None:
        return False

    try:
        expr1_simplified = sp.expand(expr1)
        expr2_simplified = sp.expand(expr2)
        return sp.simplify(expr1_simplified - expr2_simplified) == 0
    except Exception as e:
        print(f"Error simplifying or comparing expressions: {e}")
        return False

def calculate_accuracy(standard_file, model_file):
    with open(standard_file, 'r', encoding='utf-8') as f:
        standard_data = {item["question"]: item["answer"] for line in f for item in [json.loads(line)]}
    
    with open(model_file, 'r', encoding='utf-8') as f:
        model_data = {item["question"]: item["answer"] for line in f for item in [json.loads(line)]}
    
    correct = 0
    total_questions = 0
    for question, answer in model_data.items():
        if question in standard_data and compare_latex_answers(standard_data[question], answer):
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