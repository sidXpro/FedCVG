import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from json import JSONEncoder
from federated_parser import get_fedhyb_args
from torch.utils.data import DataLoader
from preprocessing.baselines_dataloader import load_data, divide_data
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch.optim as optim

class PythonObjectEncoder(JSONEncoder):    
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj )
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}
  
class FedHyb:
    def __init__(self, args):
        """初始化FedHyb算法"""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        np.random.seed(args.i_seed)
        torch.manual_seed(args.i_seed)
        random.seed(args.i_seed)
        
        # 加载数据
        self.trainset_config, self.testset = divide_data(
            num_client=args.num_client,
            dataset_name=args.dataset,
            i_seed=args.i_seed,
            distribution_type=args.distribution_type,
            alpha=args.alpha
        )
        
        # 初始化客户端和服务器
        self.init_clients()
        self.init_server()
        
        # 初始化结果记录
        self.current_phase = 1
        self.results = {
            'server': {
                'accuracy': [],
                'train_loss': [],
                'f1_score': [],
                'phase': []
            }
        }
        
        # 新增：为梯度记录和客户端选择历史初始化数据结构
        self.client_gradients = {}  # 存储客户端梯度
        self.client_last_selected = {}  # 记录每个客户端上次被选择的轮数
        for client_id in self.clients:
            self.client_gradients[client_id] = None
            self.client_last_selected[client_id] = 0  # 初始化为第0轮
        
        # 为每个客户端初始化梯度记录的衰减参数
        if hasattr(args, 'gradient_decay'):
            self.gradient_decay = args.gradient_decay
        else:
            self.gradient_decay = 0.9  # 默认衰减率
        
        # 初始化梯度阈值
        if hasattr(args, 'gradient_threshold'):
            self.gradient_threshold = args.gradient_threshold
        else:
            self.gradient_threshold = 0.01  # 默认阈值
        
        # 初始化聚类和信誉机制所需的变量
        self.use_clustering = hasattr(args, 'use_clustering') and args.use_clustering
        self.use_reputation = hasattr(args, 'use_reputation') and args.use_reputation
        
        if self.use_clustering or self.use_reputation:
            # 存储每个客户端被标记为恶意的次数
            self.malicious_counts = {client_id: 0 for client_id in self.clients}
            
            # 存储当前被认为是恶意的客户端列表
            self.blacklisted_clients = set()
            
            # 设置阈值：被标记为恶意的次数超过第一阶段训练轮数的比例
            self.reputation_threshold = args.clustering_threshold * args.phase1_rounds if hasattr(args, 'clustering_threshold') else 0.1 * args.phase1_rounds
            
            print(f"已启用{'聚类机制' if self.use_clustering else ''}{'和' if self.use_clustering and self.use_reputation else ''}{'信誉机制' if self.use_reputation else ''}")
            if self.use_reputation:
                print(f"信誉阈值设置为: {self.reputation_threshold:.1f}轮 (第一阶段总轮数的{args.clustering_threshold if hasattr(args, 'clustering_threshold') else 0.1:.1%})")
        
        # 为SCAFFOLD算法初始化控制变量（如果第一阶段使用SCAFFOLD）
        if hasattr(args, 'scaffold_for_phase1') and args.scaffold_for_phase1:
            self.control_variate = {}
            for name, param in self.server_model.state_dict().items():
                self.control_variate[name] = torch.zeros_like(param)
            
            # 为每个客户端初始化控制变量
            for client_id in self.clients:
                self.clients[client_id]['control_variate'] = {}
                for name, param in self.server_model.state_dict().items():
                    self.clients[client_id]['control_variate'][name] = torch.zeros_like(param)
    
    def init_clients(self):
        """初始化所有客户端"""
        self.clients = {}
        for client_id in self.trainset_config['users']:
            self.clients[client_id] = {
                'model': self.create_model(),
                'optimizer': None,
                'data': self.trainset_config['user_data'][client_id]
            }
    
    def init_server(self):
        """初始化服务器"""
        self.server_model = self.create_model()
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size)
    
    def create_model(self):
        """创建模型实例"""
        from models import LeNet, CNN, ResNet18, ResNet34, ResNet50, AlexCifarNet
        
        if self.args.model == 'LeNet':
            return LeNet().to(self.device)
        elif self.args.model == 'CNN':
            return CNN().to(self.device)
        elif self.args.model == 'ResNet18':
            return ResNet18().to(self.device)
        elif self.args.model == 'ResNet34':
            return ResNet34().to(self.device)
        elif self.args.model == 'ResNet50':
            return ResNet50().to(self.device)
        elif self.args.model == 'AlexCifarNet':
            return AlexCifarNet().to(self.device)
        else:
            raise NotImplementedError(f"模型 {self.args.model} 尚未实现")
    
    def freeze_feature_layers(self, model, phase):
        """
        根据训练阶段和模型类型冻结或解冻特征提取层
        
        参数:
            model: 模型实例
            phase: 训练阶段，1表示第一阶段(所有层可训练)，2表示第二阶段(冻结特征提取层)
        
        返回:
            trainable_params: 需要训练的参数列表
        """
        # 如果是第一阶段，所有参数都可训练
        if phase == 1:
            for param in model.parameters():
                param.requires_grad = True
            return model.parameters()
        
        # 在第二阶段，冻结特征提取层，只训练分类层
        trainable_params = []
        
        # 根据模型类型确定特征提取层和分类层
        if isinstance(model, type(self.server_model)):
            model_name = model.__class__.__name__
            
            # 针对不同模型进行处理
            if 'LeNet' in model_name:
                # LeNet: conv1, conv2为特征提取层；fc1, fc2, fc3为分类层
                feature_layers = ['conv1', 'conv2']
                classifier_layers = ['fc1', 'fc2', 'fc3']
                
                # 冻结特征提取层
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'CNN' in model_name:
                # CNN: conv1, conv2, conv3为特征提取层；fc1, fc2为分类层
                feature_layers = ['conv1', 'conv2', 'conv3']
                classifier_layers = ['fc1', 'fc2']
                
                # 冻结特征提取层
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'ResNet' in model_name:
                # ResNet: conv1, layer1-4为特征提取层；linear为分类层
                feature_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'bn']
                classifier_layers = ['linear']
                
                # 冻结特征提取层
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'AlexCifarNet' in model_name:
                # AlexNet: features为特征提取层；classifier为分类层
                # 冻结特征提取层
                for name, param in model.named_parameters():
                    if 'features' in name:
                        param.requires_grad = False
                    elif 'classifier' in name:
                        param.requires_grad = True
                        trainable_params.append(param)
            
            else:
                # 默认情况：只冻结前70%的层（假设前面的层是特征提取层）
                all_params = list(model.named_parameters())
                split_idx = int(len(all_params) * 0.7)
                
                for i, (name, param) in enumerate(all_params):
                    if i < split_idx:  # 前70%为特征提取层
                        param.requires_grad = False
                    else:  # 后30%为分类层
                        param.requires_grad = True
                        trainable_params.append(param)
                        
            # 在进入第二阶段时显示一次分类层信息
            if phase == 2 and self.current_phase == 1 and self.current_round == self.args.phase1_rounds - 1:
                model_state = model.state_dict()
                print(f"\n第二阶段信息: {'='*40}")
                print(f"模型: {model_name}")
                print(f"冻结特征提取层层数: {sum(1 for name, param in model.named_parameters() if not param.requires_grad)}")
                print(f"可训练分类层层数: {len(trainable_params)}")
                
                # 打印分类层的名称和维度
                print("\n分类层维度:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"  - {name}: {list(param.size())}")
                print('='*60)
                
            return trainable_params
        else:
            # 如果模型类型无法识别，则所有参数可训练
            return model.parameters()
    
    def client_update(self, client_id, phase=1):
        """
        客户端本地训练
        
        参数:
        - client_id: 客户端ID
        - phase: 训练阶段，1表示第一阶段，2表示第二阶段
        
        返回:
        - model_state: 更新后的模型状态
        - data_size: 客户端数据大小
        - epoch_loss: 客户端训练损失
        """
        client = self.clients[client_id]
        model = client['model']
        model.train()
        
        # 获取模型全局参数
        global_params = {}
        for name, param in self.server_model.named_parameters():
            global_params[name] = param.clone()
        
        # 设置学习率
        if phase == 1:
            lr = self.args.learning_rate
        else:
            lr = self.args.phase2_learning_rate
        
        # 根据当前阶段冻结或解冻特征提取层，并获取可训练参数
        trainable_params = self.freeze_feature_layers(model, phase)
        
        # 设置优化器和损失函数（只优化可训练参数）
        optimizer = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # 设置近端项系数（如果使用FedProx）
        mu = self.args.fedprox_mu if phase == 1 and not (hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1) else 0
        
        # 如果使用SCAFFOLD，准备控制变量
        if phase == 1 and hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
            # 获取当前客户端和服务器的控制变量
            client_control = client['control_variate']
            server_control = self.control_variate
            
            # 保存本地控制变量的副本用于更新
            old_client_control = {k: v.clone() for k, v in client_control.items()}
            
            # 保存参数初始值用于更新控制变量
            init_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 创建数据加载器
        train_loader = DataLoader(
            client['data'],
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        # 如果数据量很小，打印出来
        num_batches = len(train_loader)
        if num_batches < 5 and phase == 1: # 只在第一阶段显示这些信息
            print(f"客户端 {client_id} 数据较少:")
            print(f"- 样本数: {len(client['data'])}")
            print(f"- 总batch数: {num_batches}\n")
        
        # 本地训练
        epoch_loss = 0
        for epoch in range(self.args.num_local_epoch):
            batch_loss = 0
            batch_count = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                # 计算原始损失
                loss = criterion(output, target)
                
                # 第一阶段：使用FedProx或SCAFFOLD算法
                if phase == 1:
                    if hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
                        # 使用SCAFFOLD：不需要修改损失函数，但需要修改梯度
                        pass
                    else:
                        # 使用FedProx：计算近端项（proximal term）
                        proximal_term = 0.0
                        for name, param in model.named_parameters():
                            if name in global_params:
                                proximal_term += torch.sum((param - global_params[name])**2)
                        
                        # 添加近端项到损失函数
                        loss += (mu / 2) * proximal_term
                
                loss.backward()
                
                # SCAFFOLD: 第一阶段应用控制变量修正梯度
                if phase == 1 and hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in client_control and name in server_control:
                            param.grad += server_control[name] - client_control[name]
                
                # 添加梯度裁剪以防止梯度爆炸和NaN值
                max_norm = 10.0  # 最大梯度范数
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                # 检查并修复NaN梯度
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"警告: 客户端 {client_id} 在训练中出现NaN梯度，已将其置零")
                        param.grad.data = torch.zeros_like(param.grad.data)
                
                optimizer.step()
                batch_loss += loss.item()
                batch_count += 1
            
            epoch_loss = batch_loss / len(train_loader)
        
        # SCAFFOLD: 更新客户端控制变量（仅在第一阶段）
        if phase == 1 and hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
            for name, param in model.state_dict().items():
                if name in client['control_variate'] and name in init_params:
                    # 计算新的控制变量: c_i^{+} = c_i - c + (x_0 - x_K) / (K * eta)
                    lr = self.args.learning_rate
                    step_size = self.args.scaffold_c_lr / (self.args.num_local_epoch * lr)
                    client['control_variate'][name] = old_client_control[name] - server_control[name] + (init_params[name] - param) * step_size
        
        # 收集模型权重
        model_state = model.state_dict()
        
        # 在第二阶段只返回分类层的参数（可训练参数），减少通信开销
        if phase == 2:
            # 创建一个新的状态字典，只包含分类层参数
            classifier_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:  # 只包含可训练的参数（分类层）
                    classifier_state[name] = model_state[name]
            
            # 使用分类层状态代替完整模型状态
            model_state = classifier_state
        
        # 保存当前轮次的梯度（用于梯度记录机制）
        if phase == 2:  # 只在第二阶段记录梯度
            # 检查客户端是否在黑名单中
            if self.use_reputation and client_id in self.blacklisted_clients:
                # 不为黑名单客户端存储梯度
                return model_state, self.trainset_config['data_size'][idx], loss
                
            # 计算当前梯度：全局模型与本地更新后模型的差
            client_gradient = {}
            has_nan = False
            
            for name, param in model.named_parameters():
                if name in self.server_model.state_dict() and param.requires_grad:
                    server_param = self.server_model.state_dict()[name]
                    # 梯度 = 服务器参数 - 客户端更新后参数
                    grad = server_param - model_state[name]
                    
                    # 检查NaN值
                    if torch.isnan(grad).any():
                        print(f"警告: 客户端 {client_id} 的梯度包含NaN值，将不记录此梯度")
                        has_nan = True
                        break
                    
                    client_gradient[name] = grad
            
            # 只在没有NaN值时存储梯度
            if not has_nan:
                # 存储客户端梯度
                self.client_gradients[client_id] = client_gradient
                
                # 更新最后选择轮次
                self.client_last_selected[client_id] = self.current_round
            else:
                # 如果有NaN，确保清除该客户端的历史梯度
                self.client_gradients[client_id] = None
        
        # 如果启用了投毒攻击且当前客户端是恶意客户端，执行梯度投毒攻击
        if hasattr(self.args, 'enable_attack') and self.args.enable_attack:
            # 从客户端ID中提取客户端编号
            client_idx = -1
            if isinstance(client_id, str) and client_id.startswith('client_'):
                try:
                    client_idx = int(client_id.split('_')[1])
                except (IndexError, ValueError):
                    # 尝试不同的格式
                    try:
                        client_idx = int(''.join(filter(str.isdigit, client_id)))
                    except ValueError:
                        print(f"无法从客户端ID {client_id} 提取编号")
            
            # 检查客户端是否是恶意客户端
            if client_idx != -1 and client_idx < self.args.num_malicious:
                print(f"客户端 {client_id} (索引 {client_idx}) 准备执行 {self.args.attack_type} 类型的梯度投毒攻击 (阶段 {phase})")
                
                # 获取并修改模型参数
                attack_count = 0
                for name, param in model_state.items():
                    if name in self.server_model.state_dict():
                        server_param = self.server_model.state_dict()[name]
                        # 计算梯度（模型差异）
                        gradient = server_param - param
                        
                        # 根据攻击类型修改梯度
                        try:
                            if self.args.attack_type == 'gaussian':
                                # 增强gaussian攻击效果，使其更易被检测
                                # 生成随机噪声
                                noise = torch.randn_like(gradient)
                                # 标准化噪声向量
                                noise_norm = torch.norm(noise)
                                if noise_norm > 0:
                                    noise = noise / noise_norm
                                # 缩放噪声至梯度范数的倍数
                                gradient_norm = torch.norm(gradient)
                                # 使用梯度范数和指定的噪声级别确定噪声大小
                                scaling_factor = gradient_norm * max(self.args.noise_level, 3.0)
                                # 添加缩放后的噪声
                                noise = noise * scaling_factor
                                print(f"Gaussian攻击: 梯度范数={gradient_norm:.4f}, 噪声范数={scaling_factor:.4f}")
                                gradient = gradient + noise
                            elif self.args.attack_type == 'sign_flip':
                                # 翻转梯度符号，但避免极值
                                gradient = -gradient * min(self.args.noise_level, 5.0)
                            elif self.args.attack_type == 'targeted':
                                # 目标攻击：将梯度朝固定方向偏移，但限制范围
                                gradient = torch.ones_like(gradient).clamp(-5, 5) * min(self.args.noise_level, 5.0)
                            
                            # 检查修改后的梯度是否包含NaN
                            if torch.isnan(gradient).any():
                                print(f"警告: 攻击生成的梯度包含NaN值，将使用随机小值替代")
                                gradient = torch.randn_like(gradient) * 0.01  # 使用小的随机值
                            
                            # 应用梯度修改，更新模型状态
                            model_state[name] = server_param - gradient
                            attack_count += 1
                        except Exception as e:
                            print(f"执行攻击时出错: {e}，跳过此层参数")
                
                print(f"客户端 {client_id} (索引 {client_idx}) 成功执行了 {self.args.attack_type} 类型的梯度投毒攻击，修改了 {attack_count} 层参数 (阶段 {phase})")
        
        return model_state, len(client['data']), epoch_loss
    
    def server_aggregate(self, updates):
        """带梯度记录的服务器聚合模型"""
        total_weight = 0
        aggregated_weights = None
        
        # 判断是否使用SCAFFOLD进行聚合
        use_scaffold = hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1 and self.current_phase == 1
        
        # 选中客户端列表
        selected_clients = list(updates.keys())
        
        # 在第二阶段使用历史梯度
        if self.current_phase == 2:
            # 第二阶段：只聚合分类层参数（不复制整个模型状态）
            aggregated_weights = {}
            
            # 用于记录哪些层会被更新（分类层）
            updated_layers = set()
            
            # 聚合选中客户端的更新（只聚合分类层）
            for client_id, (weights, num_samples, _) in updates.items():
                total_weight += num_samples
                
                # 处理客户端提供的层（这些应该全部是分类层）
                for k in weights.keys():
                    if k not in updated_layers:
                        # 首次更新该层时初始化
                        if k not in aggregated_weights:
                            aggregated_weights[k] = torch.zeros_like(weights[k])
                        updated_layers.add(k)
                    
                    # 加权累加
                    aggregated_weights[k] += weights[k] * num_samples
            
            # 启用历史梯度虚拟聚合机制
            if hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients:
                # 获取未选中的客户端列表，并排除黑名单客户端
                unselected_clients = [client_id for client_id in self.clients.keys() 
                                     if client_id not in selected_clients 
                                     and (not self.use_reputation or client_id not in self.blacklisted_clients)]
                
                # 为未选中的客户端添加历史梯度（如果有）
                for client_id in unselected_clients:
                    if self.client_gradients[client_id] is not None:
                        # 计算衰减因子：基于轮次间隔的指数衰减
                        rounds_since_selected = self.current_round - self.client_last_selected[client_id]
                        decay_factor = self.gradient_decay ** rounds_since_selected
                        
                        # 只使用时间衰减加权，不考虑客户端数据量
                        if decay_factor > self.gradient_threshold:  # 使用类的阈值成员变量
                            # 使用统一的标准权重
                            client_effective_weight = 1.0
                            
                            # 检查梯度是否包含NaN
                            has_nan_grad = False
                            for name, grad in self.client_gradients[client_id].items():
                                if torch.isnan(grad).any():
                                    has_nan_grad = True
                                    print(f"警告: 客户端 {client_id} 的历史梯度包含NaN值，将跳过此梯度")
                                    break
                            
                            # 如果梯度包含NaN，跳过此客户端
                            if has_nan_grad:
                                # 清除此客户端的历史梯度
                                self.client_gradients[client_id] = None
                                continue
                            
                            for name, grad in self.client_gradients[client_id].items():
                                # 只处理分类层梯度（确保该层在聚合权重中）
                                if name in updated_layers:
                                    # 仅应用时间衰减，不乘以客户端数据量
                                    weighted_grad = grad * client_effective_weight * decay_factor
                                    # 添加到聚合权重中 (减去加权梯度，因为梯度 = 服务器参数 - 客户端参数)
                                    aggregated_weights[name] -= weighted_grad
                            
                            # 添加到总权重中，考虑衰减但不考虑数据量
                            total_weight += client_effective_weight * decay_factor
                            
                            if self.current_round % 20 == 0 and rounds_since_selected < 5:  # 只在特定轮次显示信息
                                print(f"客户端 {client_id} 的历史梯度被加入聚合 (衰减因子: {decay_factor:.4f})")
                                print(f"客户端 {client_id} 的历史梯度将继续保留用于后续轮次")
            
            # 计算加权平均（针对所有聚合的层）
            for k in aggregated_weights.keys():
                if total_weight > 0:
                    aggregated_weights[k] = aggregated_weights[k] / total_weight
            
            # 仅在进入第二阶段时显示一次
            if self.current_round == self.args.phase1_rounds:
                print(f"第二阶段: 只聚合分类层参数 ({len(aggregated_weights)}个层)")
                if len(aggregated_weights) > 0:
                    for layer in list(aggregated_weights.keys())[:3]:
                        print(f"  - {layer}: {aggregated_weights[layer].shape}")
                    if len(aggregated_weights) > 3:
                        print(f"  - ... 以及 {len(aggregated_weights)-3} 个其他层")
            
            return aggregated_weights
        elif use_scaffold:
            # SCAFFOLD聚合方式（第一阶段）
            client_control_sum = {}
            
            for client_id, (weights, num_samples, _) in updates.items():
                total_weight += num_samples
                # 聚合模型权重
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * num_samples for k, v in weights.items()}
                else:
                    for k in weights.keys():
                        aggregated_weights[k] += weights[k] * num_samples
                
                # 收集客户端控制变量用于更新服务器控制变量
                client_control = self.clients[client_id]['control_variate']
                if not client_control_sum:
                    client_control_sum = {k: v.clone() * num_samples for k, v in client_control.items()}
                else:
                    for k in client_control.keys():
                        client_control_sum[k] += client_control[k] * num_samples
            
            # 计算模型的加权平均
            for k in aggregated_weights.keys():
                aggregated_weights[k] = aggregated_weights[k] / total_weight
                
            # 更新服务器控制变量: c = (1/n) * sum(c_i)
            for k in client_control_sum.keys():
                self.control_variate[k] = client_control_sum[k] / total_weight
            
            return aggregated_weights
        else:
            # 标准FedAvg聚合方式（针对第一阶段或非特殊情况）
            for client_id, (weights, num_samples, _) in updates.items():
                total_weight += num_samples
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * num_samples for k, v in weights.items()}
                else:
                    for k in weights.keys():
                        aggregated_weights[k] += weights[k] * num_samples
            
            # 计算加权平均
            for k in aggregated_weights.keys():
                aggregated_weights[k] = aggregated_weights[k] / total_weight
            
            return aggregated_weights
    
    def evaluate(self):
        """评估服务器模型性能"""
        self.server_model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.server_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 收集预测和目标值用于计算F1-score
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算F1-score (多分类使用macro平均)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return correct / total, f1
    
    def detect_malicious_clients(self, updates):
        """
        使用聚类分析检测恶意客户端
        
        参数:
        - updates: 客户端更新字典，格式为 {client_id: (weights, num_samples, loss)}
        
        返回:
        - malicious_clients: 检测到的恶意客户端列表
        """
        if not self.use_clustering or self.current_phase != 1:
            return []
        
        # 提取所有客户端的更新
        client_weights = {}
        client_sample_sizes = {}
        
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
            client_sample_sizes[client_id] = num_samples
        
        # 转换模型参数为向量表示，用于聚类分析
        weight_vectors = {}
        client_ids = []
        
        # 特殊处理 - 计算与服务器模型的相对变化
        for client_id, weights in client_weights.items():
            vector = []
            # 计算与服务器模型的差异
            for name, param in weights.items():
                if name in self.server_model.state_dict():
                    server_param = self.server_model.state_dict()[name]
                    diff = server_param - param  # 计算差异
                    vector.append(diff.view(-1))
            if vector:  # 确保有差异向量
                weight_vectors[client_id] = torch.cat(vector).cpu().numpy()
                client_ids.append(client_id)
        
        # 将更新向量堆叠为矩阵，每行代表一个客户端的更新
        X = np.array([weight_vectors[client_id] for client_id in client_ids])
        
        # 客户端数量检查
        num_clients = len(client_ids)
        if num_clients < 3:  # 至少需要3个客户端才能进行有效聚类
            return []
        
        # 检查并处理NaN值
        if np.isnan(X).any():
            print(f"警告: 检测到包含NaN值的客户端更新，正在进行处理...")
            
            # 检查每个客户端的更新是否包含NaN
            nan_clients = []
            for i, client_id in enumerate(client_ids):
                if np.isnan(X[i]).any():
                    nan_clients.append(client_id)
                    print(f"客户端 {client_id} 的更新包含NaN值")
            
            # 如果所有客户端都有NaN，则无法进行聚类
            if len(nan_clients) == num_clients:
                print("错误: 所有客户端更新都包含NaN值，无法进行聚类分析")
                return nan_clients  # 将所有NaN客户端视为恶意
            
            # 处理方法1: 移除包含NaN的客户端
            valid_indices = ~np.isnan(X).any(axis=1)
            X = X[valid_indices]
            valid_client_ids = [client_ids[i] for i, valid in enumerate(valid_indices) if valid]
            
            # 如果剩余客户端太少，无法进行聚类
            if len(valid_client_ids) < 3:
                print(f"警告: 移除NaN后仅剩 {len(valid_client_ids)} 个有效客户端，直接将包含NaN的客户端标记为恶意")
                return nan_clients
            
            print(f"已移除 {len(nan_clients)} 个包含NaN的客户端，使用剩余 {len(valid_client_ids)} 个客户端进行聚类")
            client_ids = valid_client_ids
        
        # 确定聚类数量 - 增加聚类数量以更好地检测恶意客户端
        n_clusters = min(max(self.args.fedhyb_cluster_count, 3), len(client_ids) - 1)
        
        try:
            # 计算更新向量的范数 - 用于检测异常梯度
            norms = np.linalg.norm(X, axis=1)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            # 直接基于范数检测异常值
            norm_threshold = mean_norm + 2 * std_norm  # 使用2倍标准差作为阈值
            outlier_clients = []
            
            # 特别针对高斯噪声攻击的检测
            for i, client_id in enumerate(client_ids):
                if norms[i] > norm_threshold:
                    print(f"范数检测: 客户端 {client_id} 的梯度范数 {norms[i]:.4f} 超过阈值 {norm_threshold:.4f}")
                    outlier_clients.append(client_id)
            
            # 执行K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.args.i_seed).fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # 计算每个簇的大小
            cluster_sizes = {}
            for label in labels:
                if label not in cluster_sizes:
                    cluster_sizes[label] = 0
                cluster_sizes[label] += 1
            
            # 计算簇之间的距离
            cluster_distances = pairwise_distances(centers)
            
            # 降低大小阈值，使更小的簇能被识别为恶意
            size_threshold = min(self.args.fedhyb_size_threshold, 0.25)
            
            # 标识可能的恶意簇
            malicious_clusters = set()
            for i in range(n_clusters):
                # 计算簇大小比例
                size_ratio = cluster_sizes[i] / len(client_ids)
                
                # 判断是否是小簇（可能的恶意簇）
                is_small_cluster = size_ratio < size_threshold
                
                # 计算与其他簇的最小距离
                other_clusters = [j for j in range(n_clusters) if j != i]
                if other_clusters:
                    min_distance = min(cluster_distances[i][j] for j in other_clusters)
                    # 判断是否距离远离其他簇（可能的恶意簇）
                    # 降低距离阈值，使更近的簇也能被识别为异常
                    distance_threshold = self.args.fedhyb_distance_threshold * 0.75
                    is_distant_cluster = min_distance > distance_threshold
                else:
                    is_distant_cluster = False
                
                # 如果簇较小，则标记为可能的恶意簇
                if is_small_cluster:
                    malicious_clusters.add(i)
                    print(f"FedHyb聚类检测到可能的恶意簇: 簇{i}, 大小={cluster_sizes[i]}/{len(client_ids)}, 比例={size_ratio:.2f}, 最小距离={min_distance:.4f}")
            
            # 收集被标记为恶意簇的客户端
            malicious_clients = []
            for i, label in enumerate(labels):
                if label in malicious_clusters:
                    malicious_clients.append(client_ids[i])
            
            # 添加基于范数检测的异常客户端
            for client_id in outlier_clients:
                if client_id not in malicious_clients:
                    malicious_clients.append(client_id)
                    print(f"基于范数检测将客户端 {client_id} 标记为恶意")
            
            # 添加之前识别出的包含NaN的客户端
            if 'nan_clients' in locals() and nan_clients:
                print(f"将 {len(nan_clients)} 个包含NaN值的客户端标记为恶意")
                malicious_clients.extend(nan_clients)
            
            if malicious_clients:
                print(f"FedHyb第{self.current_round+1}轮检测到{len(malicious_clients)}个可能的恶意客户端: {malicious_clients}")
            
            return malicious_clients
            
        except Exception as e:
            print(f"FedHyb聚类分析过程中出错: {e}")
            # 如果聚类失败，将包含NaN的客户端标记为恶意
            if 'nan_clients' in locals() and nan_clients:
                print(f"聚类失败，将 {len(nan_clients)} 个包含NaN值的客户端标记为恶意")
                return nan_clients
            return []
    
    def update_reputation(self, malicious_clients):
        """
        更新客户端信誉系统
        
        参数:
        - malicious_clients: 当前轮次检测到的恶意客户端列表
        """
        if not self.use_reputation or self.current_phase != 1:
            return
        
        # 更新被检测为恶意的客户端计数
        for client_id in malicious_clients:
            self.malicious_counts[client_id] += 1
            
            # 如果被标记为恶意的次数超过阈值，加入黑名单
            if self.malicious_counts[client_id] >= self.reputation_threshold:
                if client_id not in self.blacklisted_clients:
                    self.blacklisted_clients.add(client_id)
                    print(f"客户端 {client_id} 被标记为恶意客户端 {self.malicious_counts[client_id]}/{self.reputation_threshold:.1f} 次，已加入黑名单")
        
        # 每10轮显示一次当前的信誉状态
        if self.current_round % 10 == 0:
            # 按恶意计数排序的客户端列表
            sorted_clients = sorted(self.malicious_counts.items(), key=lambda x: x[1], reverse=True)
            print("\n当前客户端信誉状态:")
            print(f"{'客户端ID':<10} | {'恶意计数':<10} | {'状态':<10}")
            print("-" * 35)
            
            for client_id, count in sorted_clients[:5]:  # 只显示前5个
                if count > 0:
                    status = "黑名单" if client_id in self.blacklisted_clients else "正常"
                    print(f"{client_id:<10} | {count:<10} | {status:<10}")
            
            if len(self.blacklisted_clients) > 0:
                print(f"\n当前黑名单客户端: {self.blacklisted_clients}")
    
    def select_clients(self, client_ratio):
        """根据客户端比例随机选择参与训练的客户端，排除黑名单客户端"""
        all_clients = self.trainset_config['users']
        
        # 在任何阶段都排除黑名单客户端，只要启用信誉机制
        if self.use_reputation and self.blacklisted_clients:
            eligible_clients = [client_id for client_id in all_clients if client_id not in self.blacklisted_clients]
            if not eligible_clients:  # 如果所有客户端都被拉黑（极少情况），使用所有客户端
                print("警告: 所有客户端都在黑名单中，将使用所有客户端")
                eligible_clients = all_clients
        else:
            eligible_clients = all_clients
        
        num_selected = max(1, int(len(eligible_clients) * client_ratio))
        selected_clients = random.sample(eligible_clients, num_selected)
        return selected_clients
    
    def update_model_partial(self, client_model, server_weights, phase):
        """
        根据当前阶段选择性地更新客户端模型参数
        
        参数:
            client_model: 客户端模型
            server_weights: 服务器模型权重
            phase: 当前阶段(1或2)
        """
        client_weights = client_model.state_dict()
        
        if phase == 1:
            # 第一阶段: 更新全部参数
            client_model.load_state_dict(server_weights)
        else:
            # 第二阶段: 只更新分类层参数，保持特征提取层冻结
            for name, param in client_model.named_parameters():
                if param.requires_grad and name in server_weights:
                    client_weights[name] = server_weights[name]
            
            client_model.load_state_dict(client_weights)
    
    def save_results(self):
        """保存训练结果到文件"""
        try:
            # 确保结果目录存在
            if not os.path.exists(self.args.res_root):
                os.makedirs(self.args.res_root)
                print(f"创建结果目录: {self.args.res_root}")
            
            # 修改文件名以匹配其他算法的格式
            phase1_algo = "SCAFFOLD" if (hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1) else "FedProx"
            
            # 添加附加功能到名称
            algo_suffix = ""
            if self.use_clustering:
                algo_suffix += "_Cluster"
            if self.use_reputation:
                algo_suffix += "_Rep"
            
            phase2_suffix = "_GradMem" if (hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients) else ""
            
            # 添加攻击信息到文件名
            attack_suffix = ""
            if hasattr(self.args, 'enable_attack') and self.args.enable_attack:
                attack_suffix = f"_attack_{self.args.attack_type}_{self.args.num_malicious}_{self.args.noise_level}"
            
            # 使用与其他算法相同的文件命名格式，移除.json扩展名
            filename = f'[FedHyb_{phase1_algo}{algo_suffix}{phase2_suffix}_{self.args.model}_non_iid_label_alpha{self.args.alpha}_{self.args.i_seed}]{attack_suffix}'
            result_path = os.path.join(self.args.res_root, filename)
            
            # 检查results字典是否为空
            if not self.results or not self.results.get('server') or not self.results['server'].get('accuracy'):
                print("警告: 结果数据为空，无法保存结果")
                return
            
            # 确保结果包含正确的数据
            print(f"准备保存结果数据...")
            print(f"- 准确率记录: {len(self.results['server']['accuracy'])}项")
            print(f"- 损失记录: {len(self.results['server']['train_loss'])}项")
            print(f"- F1分数记录: {len(self.results['server']['f1_score'])}项")
            
            # 使用json.dump保存结果
            with open(result_path, 'w') as f:
                json.dump(self.results, f, cls=PythonObjectEncoder)
            
            # 验证文件是否成功保存
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                print(f"结果已成功保存到: {result_path} (文件大小: {file_size/1024:.2f} KB)")
            else:
                print(f"错误: 文件保存失败，找不到文件: {result_path}")
                
        except Exception as e:
            print(f"保存结果时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试备用保存方式
            try:
                backup_path = os.path.join(self.args.res_root, f'fedhyb_results_backup_{self.args.i_seed}')
                with open(backup_path, 'w') as f:
                    json.dump(self.results, f, cls=PythonObjectEncoder)
                print(f"已创建备份结果文件: {backup_path}")
            except Exception as backup_err:
                print(f"备份保存也失败: {str(backup_err)}")
    
    def train(self):
        """FedHyb训练过程"""
        pbar = tqdm(range(self.args.total_rounds))
        max_accuracy = 0
        max_f1 = 0
        
        # 输出算法信息
        if hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
            print("=== FedHyb (SCAFFOLD -> FedAvg) 训练开始 ===")
            print(f"第一阶段 (1-{self.args.phase1_rounds}轮): SCAFFOLD算法，学习率: {self.args.learning_rate}")
            print(f"SCAFFOLD控制变量学习率: {self.args.scaffold_c_lr}")
        else:
            print("=== FedHyb (FedProx -> FedAvg) 训练开始 ===")
            print(f"第一阶段 (1-{self.args.phase1_rounds}轮): FedProx算法，学习率: {self.args.learning_rate}")
            print(f"FedProx近端项系数: {self.args.fedprox_mu}")
        
        # 显示是否启用梯度记录
        use_historical = hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients
        if use_historical:
            print(f"第二阶段 ({self.args.phase1_rounds+1}-{self.args.total_rounds}轮): 增强型FedAvg(带梯度记录)，学习率: {self.args.phase2_learning_rate}")
            print(f"梯度衰减系数: {self.gradient_decay}")
        else:
            print(f"第二阶段 ({self.args.phase1_rounds+1}-{self.args.total_rounds}轮): FedAvg算法，学习率: {self.args.phase2_learning_rate}")
        
        # 打印特征提取层冻结信息（在进入第二阶段时会有更详细的信息）
        print(f"第二阶段: 冻结特征提取层，只训练分类层")
        
        for round_idx in pbar:
            # 保存当前轮次
            self.current_round = round_idx
            
            # 确定当前阶段
            if round_idx < self.args.phase1_rounds:
                self.current_phase = 1
                # 使用客户端比例
                client_ratio = self.args.phase1_client_ratio
            else:
                # 如果是第一轮第二阶段的训练，清除黑名单客户端的历史梯度
                if round_idx == self.args.phase1_rounds:
                    self.current_phase = 2
                    # 清除黑名单客户端的历史梯度
                    if self.use_reputation and self.blacklisted_clients:
                        print(f"\n第二阶段开始：清除 {len(self.blacklisted_clients)} 个黑名单客户端的历史梯度")
                        for client_id in self.blacklisted_clients:
                            if client_id in self.client_gradients:
                                self.client_gradients[client_id] = None
                                print(f"已清除客户端 {client_id} 的历史梯度")
                else:
                    self.current_phase = 2
                # 使用客户端比例
                client_ratio = self.args.phase2_client_ratio
            
            # 选择客户端
            selected_clients = self.select_clients(client_ratio)
            
            # 客户端本地训练
            client_updates = {}
            for client_id in selected_clients:
                # 更新客户端模型（根据阶段决定是全部更新还是部分更新）
                self.update_model_partial(self.clients[client_id]['model'], self.server_model.state_dict(), self.current_phase)
                
                # 本地训练
                weights, num_samples, loss = self.client_update(client_id, phase=self.current_phase)
                client_updates[client_id] = (weights, num_samples, loss)
            
            # 在第一阶段启用聚类检测和信誉机制
            if self.current_phase == 1 and (self.use_clustering or self.use_reputation):
                # 检测恶意客户端
                malicious_clients = self.detect_malicious_clients(client_updates)
                
                # 更新客户端信誉
                if self.use_reputation:
                    self.update_reputation(malicious_clients)
                
                # 如果检测到恶意客户端，从更新中移除（但保留在信誉系统中的记录）
                if malicious_clients:
                    print(f"第{self.current_round+1}轮: 从聚合中排除 {len(malicious_clients)} 个恶意客户端")
                    for client_id in malicious_clients:
                        if client_id in client_updates:
                            del client_updates[client_id]
                            print(f"  - 已排除客户端 {client_id}")
                    
                    if len(client_updates) == 0:
                        print("警告: 所有参与的客户端都被检测为恶意，恢复使用所有客户端")
                        client_updates = {}
                        for client_id in selected_clients:
                            weights, num_samples, loss = self.client_update(client_id, phase=self.current_phase)
                            client_updates[client_id] = (weights, num_samples, loss)
            
            # 服务器聚合
            aggregated_weights = self.server_aggregate(client_updates)
            
            # 检查聚合权重是否包含NaN
            has_nan_weights = False
            if aggregated_weights:
                for k in aggregated_weights:
                    if torch.isnan(aggregated_weights[k]).any():
                        print(f"警告: 聚合后的权重包含NaN值，将跳过此次更新")
                        has_nan_weights = True
                        break
            
            # 仅在没有NaN时更新服务器模型
            if not has_nan_weights and aggregated_weights:
                # 更新服务器模型
                if self.current_phase == 2:
                    # 第二阶段：仅更新分类层参数
                    current_state = self.server_model.state_dict()
                    for k in aggregated_weights:
                        if k in current_state:
                            current_state[k] = aggregated_weights[k]
                    self.server_model.load_state_dict(current_state)
                else:
                    # 第一阶段：更新整个模型
                    self.server_model.load_state_dict(aggregated_weights)
            
            # 评估
            accuracy, f1 = self.evaluate()
            avg_loss = sum(update[2] for update in client_updates.values()) / len(client_updates) if client_updates else 0
            
            # 记录结果
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['train_loss'].append(avg_loss)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['phase'].append(self.current_phase)
            
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            if f1 > max_f1:
                max_f1 = f1
            
            # 显示当前使用的算法
            algorithm_name = ""
            if self.current_phase == 1:
                algorithm_name = "SCAFFOLD" if (hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1) else "FedProx"
                if self.use_clustering:
                    algorithm_name += "+聚类"
                if self.use_reputation:
                    algorithm_name += "+信誉机制"
            else:
                algorithm_name = "增强型FedAvg(带梯度记录)" if use_historical else "FedAvg"
            
            # 添加客户端选择信息
            client_info = f'选中: {len(selected_clients)}/{len(self.trainset_config["users"])}'
            if self.current_phase == 1 and self.use_reputation and self.blacklisted_clients:
                client_info += f' (黑名单: {len(self.blacklisted_clients)})'
                
            if self.current_phase == 2 and use_historical and self.current_round % 10 == 0:  # 每10轮才显示历史梯度信息
                # 计算有效参与的未选中客户端数量，排除黑名单客户端
                effective_unselected = 0
                for client_id in self.trainset_config["users"]:
                    if client_id not in selected_clients and self.client_gradients.get(client_id) is not None:
                        # 检查客户端是否在黑名单中
                        if self.use_reputation and client_id in self.blacklisted_clients:
                            continue
                        rounds_since = self.current_round - self.client_last_selected[client_id]
                        if self.gradient_decay ** rounds_since > 0.01:
                            effective_unselected += 1
                
                if effective_unselected > 0:
                    client_info += f' (历史梯度: +{effective_unselected})'
                
                # 在第二阶段也显示黑名单信息
                if self.use_reputation and self.blacklisted_clients:
                    client_info += f' (黑名单: {len(self.blacklisted_clients)})'
            
            # 更新进度条信息
            pbar.set_description(
                f"轮次: {round_idx+1}/{self.args.total_rounds} | "
                f"阶段: {self.current_phase} | "
                f"算法: {algorithm_name} | "
                f"准确率: {accuracy:.4f} | "
                f"F1: {f1:.4f} | "
                f"{client_info}"
            )
        
        # 保存最终结果
        print(f"\nFedHyb训练完成")
        print(f"最高准确率: {max_accuracy:.4f}")
        print(f"最高F1分数: {max_f1:.4f}")
        
        # 保存结果到文件
        self.save_results()
        
        return self.results

def main():
    args = get_fedhyb_args()
    fedhyb = FedHyb(args)
    fedhyb.train()

if __name__ == "__main__":
    main() 