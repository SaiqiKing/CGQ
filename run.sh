#!/bin/bash

# 定义要执行的命令列表
commands=(
"export HF_ENDPOINT=https://hf-mirror.com"
"python quant.py"
)

# 遍历命令列表并执行  
for cmd in "${commands[@]}"; do  
    echo "正在执行: $cmd"  
    eval "$cmd"  
    # 如果需要检查上一个命令是否成功，可以添加以下行  
    if [ $? -ne 0 ]; then  
        echo "命令执行失败，退出脚本"  
        exit 1
    fi  
done
echo "所有命令执行完毕"
