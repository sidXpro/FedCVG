# -*- coding: utf-8 -*-

"""
FedHyb算法主程序入口
"""

import os
import torch
import random
import numpy as np
from fedcvg import FedHyb
from federated_parser import get_fedhyb_args

def set_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置OMP_NUM_THREADS=1以避免KMeans在Windows上的内存泄漏问题
    os.environ['OMP_NUM_THREADS'] = '1'

def main():
    """主函数"""
    # 获取命令行参数
    args = get_fedhyb_args()
    
    # 设置随机种子
    set_seed(args.i_seed)
    
    # 打印实验配置
    print("\n" + "="*50)
    print("FedHyb实验配置:")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"数据分布类型: {args.distribution_type}")
    print(f"客户端数量: {args.num_client}")
    print(f"第一阶段轮数: {args.phase1_rounds}")
    print(f"第二阶段轮数: {args.phase2_rounds}")
    print(f"总轮数: {args.total_rounds}")
    print(f"第一阶段客户端比例: {args.phase1_client_ratio}")
    print(f"第二阶段客户端比例: {args.phase2_client_ratio}")
    print(f"第二阶段学习率: {args.phase2_learning_rate}")
    print(f"第一阶段算法: {'SCAFFOLD' if args.scaffold_for_phase1 else 'FedProx'}")
    print("="*50 + "\n")
    
    # 创建结果目录
    if not os.path.exists(args.res_root):
        os.makedirs(args.res_root)
    
    # 初始化并运行FedHyb
    fedhyb = FedHyb(args)
    results = fedhyb.train()
    
    print("\n" + "="*50)
    print("FedHyb训练完成!")
    print(f"结果保存目录: {os.path.abspath(args.res_root)}")
    
    # 列出结果目录中的文件
    try:
        result_files = os.listdir(args.res_root)
        fedhyb_files = [f for f in result_files if f.startswith('[FedHyb_')]
        
        if fedhyb_files:
            print(f"\n最新的FedHyb结果文件:")
            # 按修改时间排序，显示最新的文件
            sorted_files = sorted(fedhyb_files, 
                                  key=lambda x: os.path.getmtime(os.path.join(args.res_root, x)),
                                  reverse=True)
            
            for i, file in enumerate(sorted_files[:3]):  # 只显示最新的3个文件
                file_path = os.path.join(args.res_root, file)
                file_time = os.path.getmtime(file_path)
                file_size = os.path.getsize(file_path)
                import time
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
                print(f"{i+1}. {file} ({file_size/1024:.1f} KB, {time_str})")
                
            if len(sorted_files) > 3:
                print(f"...以及其他 {len(sorted_files)-3} 个FedHyb结果文件")
        else:
            print("警告: 未找到FedHyb结果文件!")
    except Exception as e:
        print(f"列出结果文件时出错: {str(e)}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 