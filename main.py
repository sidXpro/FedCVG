from federated_parser import get_args
from federated_learning import FederatedLearning
import os

def main():
    """
    联邦学习主程序
    运行示例：
    python main.py --dataset MNIST --model LeNet --num_client 10 --num_local_class 2 --num_round 100
    """
    # 设置OMP_NUM_THREADS=1以避免KMeans在Windows上的内存泄漏问题
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 获取参数配置
    args = get_args()
    
    # 创建并运行联邦学习实例
    fed_learning = FederatedLearning(args)
    fed_learning.train()

if __name__ == "__main__":
    main() 