#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从预训练模型恢复训练的脚本
使用方法: python resume_train.py --task=limx_arm
"""

import os
import sys
sys.path.append('/home/ril/myq/tron1-rl-isaacgym-master')

from legged_gym.scripts.train import train
from legged_gym.utils.helpers import get_args

def main():
    # 获取命令行参数
    args = get_args()
    
    # 设置任务名称（如果没有通过命令行指定）
    if args.task is None:
        args.task = "limx_arm"
    
    print(f"开始从预训练模型恢复训练...")
    print(f"任务: {args.task}")
    print(f"恢复训练: {args.resume}")
    print(f"加载运行: {args.load_run}")
    print(f"检查点: {args.checkpoint}")
    
    # 开始训练
    storage = train(args)
    
    print("训练完成!")

if __name__ == "__main__":
    main()
