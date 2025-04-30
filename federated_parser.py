import argparse

def get_args():
    """
    获取联邦学习所需的所有参数配置，包括基础算法和FedHyb特有参数
    """
    parser = argparse.ArgumentParser(description='联邦学习参数配置')
    
    # 系统相关参数
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100'],
                        help='数据集选择')
    parser.add_argument('--model', type=str, default='LeNet',
                        choices=["LeNet", 'CNN', 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50"],
                        help='模型选择')
    parser.add_argument('--num_client', type=int, default=10, help='客户端数量')
    parser.add_argument('--client_ratio', type=float, default=0.5, help='每轮参与联邦学习的客户端比例，范围[0,1]')
    parser.add_argument('--num_round', type=int, default=100, help='联邦学习轮数')
    parser.add_argument('--i_seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--res_root', type=str, default='./results', help='结果保存路径')
    
    # 数据分布相关参数
    parser.add_argument('--distribution_type', type=str, default='non_iid_label',
                        choices=['iid', 'non_iid_label'],
                        help='数据分布类型: iid(独立同分布), non_iid_label(使用狄利克雷分布控制标签倾斜)')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='狄利克雷分布参数alpha，控制标签倾斜程度，越小越不平衡')
    
    # 通用客户端训练参数
    parser.add_argument('--fed_algo', type=str, default='MSGuard',
                        choices=["FedAvg", "SCAFFOLD", "FedProx", "FedNova", "FedHyb", 
                                "Krum", "MultiKrum", "Bulyan", "TrimmedMean", "Median", "Auror", "SecDefender", "MSGuard"],
                        help='联邦学习算法选择')
    parser.add_argument('--num_local_epoch', type=int, default=3, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量因子')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减系数')
    
    # MSGuard特有参数
    parser.add_argument('--msguard_norm_lower', type=float, default=0.1, 
                        help='MSGuard范数筛选下界L，范数小于此值的梯度将被过滤，仅当fed_algo=MSGuard时有效')
    parser.add_argument('--msguard_norm_upper', type=float, default=10.0, 
                        help='MSGuard范数筛选上界R，范数大于此值的梯度将被过滤，仅当fed_algo=MSGuard时有效')
    parser.add_argument('--msguard_sign_sample_ratio', type=float, default=0.3, 
                        help='MSGuard符号统计采样比例β，值越大计算越精确但效率越低，范围(0,1]，仅当fed_algo=MSGuard时有效')
    parser.add_argument('--msguard_svd_dim', type=int, default=20, 
                        help='MSGuard谱分析采样维度ζ，维度越高计算越精确但效率越低，仅当fed_algo=MSGuard时有效')
    parser.add_argument('--msguard_bandwidth', type=float, default=None, 
                        help='MSGuard聚类的带宽参数，若为None则自动估计，仅当fed_algo=MSGuard时有效')
    
    # FedProx特有参数
    parser.add_argument('--fedprox_mu', type=float, default=0.001, 
                        help='FedProx算法的近端项系数，控制本地模型与全局模型的偏离程度，仅当fed_algo=FedProx或FedHyb时有效')
    
    # SCAFFOLD特有参数
    parser.add_argument('--scaffold_c_lr', type=float, default=0.001, 
                        help='SCAFFOLD算法的控制变量学习率，仅当fed_algo=SCAFFOLD时有效')
    
    # SecDefender特有参数
    parser.add_argument('--use_auxiliary_dataset', type=bool, default=True,
                        help='SecDefender中是否使用辅助验证集对客户端模型进行评估，仅当fed_algo=SecDefender时有效')
    parser.add_argument('--auxiliary_dataset_size', type=float, default=0.001, 
                        help='辅助验证集大小占训练集比例，范围(0,1]，仅当fed_algo=SecDefender且use_auxiliary_dataset=True时有效')
    parser.add_argument('--max_exclude_count', type=int, default=3, 
                        help='SecDefender中每轮最大剔除低质量模型的数量阈值τ，仅当fed_algo=SecDefender时有效')
    parser.add_argument('--model_quality_metric', type=str, default='f1_score',
                        choices=['f1_score', 'accuracy', 'precision', 'recall', 'balanced_accuracy'],
                        help='SecDefender中评估模型质量的指标，仅当fed_algo=SecDefender时有效')
    
    # FedNova特有参数
    parser.add_argument('--fednova_tau_eff', type=str, default='uniform', 
                        choices=['uniform', 'n_local_epoch'], 
                        help='FedNova中归一化权重的计算方式，uniform表示统一权重，n_local_epoch表示按本地迭代次数加权，仅当fed_algo=FedNova时有效')
    
    # FedHyb特有参数 - 两阶段训练
    parser.add_argument('--phase1_rounds', type=int, default=40, 
                        help='第一阶段训练轮数，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--phase2_rounds', type=int, default=60, 
                        help='第二阶段训练轮数，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--phase1_client_ratio', type=float, default=0.5, 
                        help='第一阶段每轮参与训练的客户端比例，范围[0,1]，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--phase2_client_ratio', type=float, default=0.5, 
                        help='第二阶段每轮参与训练的客户端比例，范围[0,1]，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--phase2_learning_rate', type=float, default=0.01, 
                        help='第二阶段的学习率，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--scaffold_for_phase1', type=bool, default=True,
                        help='在FedHyb第一阶段使用SCAFFOLD算法而不是FedProx，仅当fed_algo=FedHyb时有效')
    
    # FedHyb聚类和信誉机制参数
    parser.add_argument('--use_clustering', type=bool, default=True,
                        help='在FedHyb第一阶段是否使用聚类机制检测恶意客户端，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--use_reputation', type=bool, default=True,
                        help='在FedHyb第一阶段是否使用信誉机制管理客户端参与，仅当fed_algo=FedHyb时有效')
    parser.add_argument('--clustering_threshold', type=float, default=0.25,
                        help='信誉机制中，被标记为恶意的次数占训练轮数的比例阈值，超过此阈值将被永久排除，仅当use_reputation=True时有效')
    parser.add_argument('--fedhyb_cluster_count', type=int, default=2, 
                        help='FedHyb聚类算法中的簇数量，通常设置为2（正常簇和恶意簇），仅当use_clustering=True时有效')
    parser.add_argument('--fedhyb_distance_threshold', type=float, default=1, 
                        help='FedHyb聚类算法中判断异常簇的距离阈值，若簇间距离超过此阈值则认为是恶意簇，仅当use_clustering=True时有效')
    parser.add_argument('--fedhyb_size_threshold', type=float, default=0.45, 
                        help='FedHyb聚类算法中判断异常簇的大小阈值，若簇大小比例小于此阈值则可能是恶意簇，仅当use_clustering=True时有效')
    
    # Krum / MultiKrum / Bulyan特有参数
    parser.add_argument('--num_malicious_tolerance', type=int, default=2, 
                        help='鲁棒算法能够容忍的最大恶意客户端数量，仅当fed_algo=Krum、MultiKrum或Bulyan时有效')
    parser.add_argument('--multikrum_k', type=int, default=1, 
                        help='MultiKrum算法选择的客户端数量，或Bulyan第一阶段选择的客户端数量，仅当fed_algo=MultiKrum或Bulyan时有效')
    parser.add_argument('--use_distances_as_weights', type=bool, default=False, 
                        help='是否将距离用作权重进行加权聚合（而非简单选取最佳客户端），仅当fed_algo=Krum或MultiKrum时有效')
    parser.add_argument('--bulyan_beta', type=float, default=1.0,
                        help='Bulyan修剪度参数，用于确定去除多少极端值，仅当fed_algo=Bulyan时有效')
    
    # TrimmedMean / Median特有参数
    parser.add_argument('--trimmed_ratio', type=float, default=0.4, 
                        help='修剪均值算法修剪比例，表示从每个坐标上修剪多少比例的最大值和最小值，范围[0, 0.5)，仅当fed_algo=TrimmedMean时有效')
    parser.add_argument('--trim_k', type=int, default=None, 
                        help='修剪均值算法直接指定修剪数量，设置后优先于trimmed_ratio，表示从每个坐标的两端各修剪掉k个值，仅当fed_algo=TrimmedMean时有效')
    
    # Auror特有参数
    parser.add_argument('--auror_n_clusters', type=int, default=2, 
                        help='Auror聚类算法中的簇数量，通常设置为2（正常簇和恶意簇），仅当fed_algo=Auror时有效')
    parser.add_argument('--auror_distance_threshold', type=float, default=0.5, 
                        help='Auror判断异常簇的距离阈值，若簇间距离超过此阈值则认为是恶意簇，仅当fed_algo=Auror时有效')
    parser.add_argument('--auror_size_threshold', type=float, default=0.45, 
                        help='Auror判断异常簇的大小阈值，若簇大小比例小于此阈值则可能是恶意簇，仅当fed_algo=Auror时有效')
    
    # 投毒攻击相关参数
    parser.add_argument('--enable_attack', type=bool, default=False,
                        help='是否启用梯度投毒攻击')
    parser.add_argument('--num_malicious', type=int, default=2,
                        help='恶意客户端数量，从客户端0到num_malicious-1的客户端会进行投毒攻击')
    parser.add_argument('--attack_type', type=str, default='sign_flip',
                        choices=['gaussian', 'sign_flip', 'targeted'],
                        help='攻击类型：gaussian(添加高斯噪声)、sign_flip(翻转梯度符号)、targeted(目标攻击)')
    parser.add_argument('--noise_level', type=float, default=1.0,
                        help='噪声水平，控制添加噪声的强度')
    
    # 梯度记录机制参数
    parser.add_argument('--use_historical_gradients', action='store_true', help='第二阶段是否使用历史梯度记录机制')
    parser.add_argument('--gradient_decay', type=float, default=3, help='历史梯度衰减系数，范围(0,1]')
    parser.add_argument('--gradient_threshold', type=float, default=0.2, help='历史梯度使用的阈值，小于此值时不使用该历史梯度')
    
    args = parser.parse_args()
    
    # 计算FedHyb总训练轮数
    if args.fed_algo == 'FedHyb':
        args.total_rounds = args.phase1_rounds + args.phase2_rounds
    else:
        args.total_rounds = args.num_round
    
    return args

def get_fedhyb_args():
    """
    获取FedHyb算法的命令行参数（向前兼容）
    
    返回:
        args: 解析后的参数
    """
    args = get_args()
    # 确保算法为FedHyb
    args.fed_algo = 'FedHyb'
    # 计算总训练轮数
    args.total_rounds = args.phase1_rounds + args.phase2_rounds
    
    # 根据第一阶段算法确保相关参数设置正确
    if hasattr(args, 'scaffold_for_phase1') and args.scaffold_for_phase1:
        # 确保SCAFFOLD相关参数已设置
        if not hasattr(args, 'scaffold_c_lr') or args.scaffold_c_lr is None:
            args.scaffold_c_lr = 0.001  # 使用默认值
    else:
        # 确保FedProx相关参数已设置
        if not hasattr(args, 'fedprox_mu') or args.fedprox_mu is None:
            args.fedprox_mu = 0.01  # 使用默认值
    
    # 确保第二阶段学习率设置正确
    if not hasattr(args, 'phase2_learning_rate') or args.phase2_learning_rate is None:
        args.phase2_learning_rate = 0.001  # 使用默认值
    
    # 确保聚类和信誉机制参数设置正确
    if hasattr(args, 'use_clustering') and args.use_clustering:
        # 确保聚类相关参数已设置
        if not hasattr(args, 'fedhyb_cluster_count'):
            args.fedhyb_cluster_count = 2
        if not hasattr(args, 'fedhyb_distance_threshold'):
            args.fedhyb_distance_threshold = 0.5
        if not hasattr(args, 'fedhyb_size_threshold'):
            args.fedhyb_size_threshold = 0.45
    
    if hasattr(args, 'use_reputation') and args.use_reputation:
        # 确保信誉机制参数已设置
        if not hasattr(args, 'clustering_threshold'):
            args.clustering_threshold = 0.1
    
    return args

if __name__ == "__main__":
    args = get_args()
    print(args) 