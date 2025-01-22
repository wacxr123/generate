import json
import sys
import sympy as sp
from sympy.parsing.latex import parse_latex
def parse_latex_e(latex_str):
    try:
        expr = parse_latex(latex_str)
    except LaTeXParsingError as e:
        print(f"LaTeX parsing error: {e}")
        return None
    try:
        
        expr = sp.sympify(expr, evaluate=True)
        return expr
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None

def compare_latex_answers(latex1, latex2):
    expr1 = parse_latex_e(latex1)
    expr2 = parse_latex_e(latex2)

    if expr1 is None or expr2 is None:
        return False

    expr1_simplified = sp.expand(expr1)  
    expr2_simplified = sp.expand(expr2)  

    return expr1_simplified == expr2_simplified

def calculate_accuracy(standard_file, model_file):
    with open(standard_file, 'r', encoding='utf-8') as f:
        standard_data = {item["question"]: item["answer"] for item in json.load(f)}
    with open(model_file, 'r', encoding='utf-8') as f:
        model_data = {item["question"]: item["answer"] for item in json.load(f)}
    
    correct = 0
    for question, answer in model_data.items():
        if question in standard_data and compare_latex_answers(standard_data[question], answer):
            correct += 1
    
    return correct / len(standard_data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_output_jsonl.py <standard_file> <model_file>")
        sys.exit(1)
    standard_file = sys.argv[1]
    model_file = sys.argv[2]
    accuracy = calculate_accuracy(standard_file, model_file)
    print("Accuracy:", accuracy)

