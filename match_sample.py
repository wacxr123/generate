import json
import argparse
from typing import Dict, Set

def extract_matching_records(file_a: str, file_b: str, output_file: str) -> None:
    """
    提取文件A中与文件B的question字段匹配的记录
    
    参数:
    file_a (str): 第一个输入JSONL文件路径（需要提取的源文件）
    file_b (str): 第二个输入JSONL文件路径（用于匹配的文件）
    output_file (str): 输出JSONL文件路径
    """
    # 首先读取文件B中的所有question，存入集合中以便快速查找
    questions_b = set()
    with open(file_b, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if 'question' in record:
                    questions_b.add(record['question'])
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line.strip()}")
    
    print(f"从文件B中读取了 {len(questions_b)} 个问题")
    
    # 读取文件A并检查匹配
    matched_count = 0
    with open(file_a, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                record = json.loads(line.strip())
                if 'question' in record and record['question'] in questions_b:
                    # 找到匹配，写入输出文件
                    f_out.write(line)
                    matched_count += 1
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line.strip()}")
    
    print(f"处理完成！共找到 {matched_count} 条匹配记录")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='从两个JSONL文件中提取匹配的记录')
    parser.add_argument('file_a', help='源JSONL文件路径')
    parser.add_argument('file_b', help='用于匹配的JSONL文件路径')
    parser.add_argument('-o', '--output', default='output.jsonl',
                        help='输出文件路径 (默认: output.jsonl)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行匹配操作
    extract_matching_records(args.file_a, args.file_b, args.output)

if __name__ == "__main__":
    main()
