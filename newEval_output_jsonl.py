import json
import sys
import sympy as sp
from sympy.parsing.latex import parse_latex

def parse_latex_e(latex_str):
    if not latex_str.strip():  # Handle empty or whitespace-only strings
        return None
    try:
        # Pre-process percentage notation
        latex_str = latex_str.replace(r'\%', r'/100')

        expr = parse_latex(latex_str)
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
    standard_data = {}
    with open(standard_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            standard_data[item["question"]] = item["answer"]

    model_data = {}
    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            model_data[item["question"]] = item["answer"]
    correct = 0
    total = 0
    for question, answer in standard_data.items():
        total += 1
        if question in model_data:
            if compare_latex_answers(answer, model_data[question]):
                correct += 1

    if total == 0:
        return 0.0
    return correct / total

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_output_jsonl.py <standard_file> <model_file>")
        sys.exit(1)
    standard_file = sys.argv[1]
    model_file = sys.argv[2]
    accuracy = calculate_accuracy(standard_file, model_file)
    print("Accuracy:", accuracy)