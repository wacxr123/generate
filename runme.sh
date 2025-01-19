#!/bin/bash

# 循环运行每个 part
for i in {0..3}; do
    # 运行 nohup 并将输出重定向到不同的文件
    nohup python rawLLM_generator.py --device "$i"  --part "$i" > "output_part_$i.log" 2>&1 & disown
done

