import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import math

# --- SReC 核心组件实现 ---

class PixelPredictorCNN(nn.Module):
    """
    简化的像素预测CNN。
    输入：上下文张量 (batch_size, in_channels, H_context, W_context)
    输出：K个混合权重、K个位置参数（均值）和K个尺度参数，
          形状为 (batch_size, K, 1, 1)
    """
    def __init__(self, in_channels, K=10):
        super(PixelPredictorCNN, self).__init__()
        self.K = K
        # 修正1: 调整卷积层以确保最终输出的 H, W 为 1
        # 假设 H_context, W_context 为 3 (来自 3x3 窗口)
        # kernel_size=3, padding=0 会将 3x3 降为 1x1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=0), # Input (B, C_in, 3, 3) -> Output (B, 64, 1, 1)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1), # Spatial dims remain 1x1
            nn.ReLU()
        )
        # 输出 K*3 个参数，空间维度为 1x1
        self.param_predictor = nn.Conv2d(32, K * 3, kernel_size=1) # Spatial dims remain 1x1

    def forward(self, context):
        # context 形状应为 (batch_size, in_channels, H_context, W_context)
        # 期望 H_context, W_context 至少为 3
        features = self.conv_layers(context) # Output shape: (batch_size, 32, 1, 1)
        params = self.param_predictor(features) # Output shape: (batch_size, K*3, 1, 1)
        
        # Reshape params to (batch_size, K, 3, 1, 1)
        params_reshaped = params.view(params.size(0), self.K, 3, params.size(2), params.size(3))
        
        # 提取权重、均值、尺度。它们的形状将是 (batch_size, K, 1, 1)
        weights_logits = params_reshaped[:, :, 0, :, :] 
        means = params_reshaped[:, :, 1, :, :]         
        scales = torch.exp(params_reshaped[:, :, 2, :, :]) # 确保尺度为正
        # 修正4: 限制 scales 的最大值，防止数值溢出
        scales = torch.clamp(scales, max=1e5) 

        return weights_logits, means, scales

def logistic_cdf(x, mu, s):
    """
    计算离散逻辑分布的CDF（累积分布函数）差值，用于表示像素值概率。
    论文中定义为 σ((x−µ+0.5)/s)−σ((x+µ+0.5)/s)
    这里我们计算 P(X <= x) - P(X <= x-1)，即 P(X=x)
    """
    # 确保尺度s不是0，避免除以零
    s = torch.clamp(s, min=1e-5)
    
    # Sigmoid函数
    sigmoid = torch.sigmoid
    
    # P(X <= x)
    cdf_upper = sigmoid((x - mu + 0.5) / s)
    # P(X <= x-1)
    cdf_lower = sigmoid((x - mu - 0.5) / s)
    
    return cdf_upper - cdf_lower

def mixture_log_prob(target_pixel_value, weights_logits, means, scales):
    """
    计算给定目标像素值在混合逻辑分布下的对数概率。
    target_pixel_value: 目标像素值 (batch_size, H, W)
    weights_logits: 混合权重对数 (batch_size, K, H, W)
    means: 混合均值 (batch_size, K, H, W)
    scales: 混合尺度 (batch_size, K, H, W)
    """
    batch_size, K, H, W = weights_logits.shape
    
    # 归一化权重
    weights = torch.softmax(weights_logits, dim=1) # (batch_size, K, H, W)
    
    # 扩展 target_pixel_value 以匹配混合分布的维度
    target_pixel_value_expanded = target_pixel_value.unsqueeze(1).expand(-1, K, -1, -1) # (batch_size, K, H, W)
    
    # 计算每个分量的概率 P(x=k|mu,s)
    component_probs = logistic_cdf(target_pixel_value_expanded, means, scales) # (batch_size, K, H, W)
    
    # 混合概率 P(x) = sum(w_k * P_k(x))
    mixture_probs = torch.sum(weights * component_probs, dim=1) # (batch_size, H, W)
    
    # 避免log(0)
    mixture_probs = torch.clamp(mixture_probs, min=1e-10)
    
    return torch.log(mixture_probs) # (batch_size, H, W)

def mixture_entropy(weights_logits, means, scales, pixel_range=256):
    """
    计算混合逻辑分布的熵。
    这是一个近似计算，因为需要对所有可能的像素值 (0-255) 进行求和。
    在实际训练中，通常会使用 NLL 作为损失函数，熵则用于评估。
    """
    batch_size, K, H, W = weights_logits.shape
    
    weights = torch.softmax(weights_logits, dim=1) # (batch_size, K, H, W)
    
    # 创建所有可能的像素值张量
    all_pixel_values = torch.arange(0, pixel_range, device=means.device, dtype=means.dtype) # (pixel_range,)
    
    # 扩展维度以进行广播计算
    all_pixel_values_expanded = all_pixel_values.view(1, 1, pixel_range, 1, 1).expand(batch_size, K, -1, H, W)
    means_expanded = means.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    scales_expanded = scales.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    weights_expanded = weights.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    
    # 计算每个分量在所有像素值上的概率 P(x=k|mu,s)
    component_probs_all_k = logistic_cdf(all_pixel_values_expanded, means_expanded, scales_expanded) # (batch_size, K, pixel_range, H, W)
    
    # 混合概率 P(x) = sum(w_k * P_k(x))
    mixture_probs_all_k = torch.sum(weights_expanded * component_probs_all_k, dim=1) # (batch_size, pixel_range, H, W)
    
    # 避免log(0)
    mixture_probs_all_k = torch.clamp(mixture_probs_all_k, min=1e-10)
    
    # 计算熵 H = -sum(P(x) * log(P(x)))
    entropy_map = -torch.sum(mixture_probs_all_k * torch.log(mixture_probs_all_k), dim=1) # (batch_size, H, W)
    
    return entropy_map

# --- 图像预处理和多分辨率生成 ---

def _2x2_avg_pool(image_tensor):
    """
    对图像张量进行 2x2 平均池化。
    image_tensor: (C, H, W)
    返回: (C, H/2, W/2)
    """
    # 使用 nn.AvgPool2d 进行池化，并确保输出是浮点数
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    # 增加 batch 维度，进行池化，然后移除 batch 维度
    return pool(image_tensor.unsqueeze(0)).squeeze(0)

def _round_to_int(image_tensor):
    """
    将浮点图像张量四舍五入并转换为整数（0-255）。
    """
    return torch.round(image_tensor).clamp(0, 255).to(torch.float32) # 保持float以便后续计算

def generate_multi_resolution_images(image_tensor):
    """
    生成图像的多分辨率版本 x(l) 和 y(l)。
    image_tensor: 原始图像 (C, H, W), 范围 0-255
    返回: x_levels (list of tensors), y_levels (list of tensors)
    """
    x_levels = [image_tensor] # x(0) = 原始图像
    y_levels = []

    current_x = image_tensor
    for l in range(3): # 生成 y(1), y(2), y(3) 和 x(1), x(2), x(3)
        # y(l+1) = avpool(x(l))
        y_next = _2x2_avg_pool(current_x)
        y_levels.append(y_next)
        
        # x(l+1) = round(y(l+1))
        x_next = _round_to_int(y_next)
        x_levels.append(x_next)
        current_x = x_next
    
    return x_levels, y_levels

def get_pixel_context(x_levels, y_levels, level, i, j, context_window_size=3):
    """
    获取像素 (i,j) 在指定级别 l 的上下文 X(l)i,j。
    上下文包括低分辨率图像 y(l+1) 的一部分和同分辨率邻居。
    为了简化，我们使用 y(l+1) 中对应的单个像素作为低分辨率上下文，
    并使用当前级别 x(l) 的 3x3 邻居（不包括中心像素）作为同分辨率上下文。
    """
    current_x = x_levels[level] # x(l)
    C, H, W = current_x.shape

    context_features = []

    # 1. 低分辨率图像 y(l+1) 的对应像素
    if level < 3: # 只有 levels 0, 1, 2 需要 y(l+1)
        # 对应 y(l+1) 中的像素 (i//2, j//2)
        low_res_y_pixel = y_levels[level][:, i // 2, j // 2].unsqueeze(-1).unsqueeze(-1) # (C, 1, 1)
        context_features.append(low_res_y_pixel)
    else:
        # x(3) 是最低分辨率，没有 y(4) 作为上下文。
        # 论文提到 x(3) 是“明文编码”，这里我们只为 levels 0,1,2 提取上下文
        pass 

    # 2. 同分辨率邻居 (3x3 窗口)
    # 创建一个填充后的张量以处理边界
    pad_size = context_window_size // 2
    padded_x = torch.nn.functional.pad(current_x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # 提取 3x3 窗口，中心是 (i+pad_size, j+pad_size)
    window = padded_x[:, i:i+context_window_size, j:j+context_window_size] # (C, 3, 3)
    
    context_features.append(window)

    # 修正2: 确保 context_tensor 的通道数与 PixelPredictorCNN 期望的一致
    # 这里我们是 cat(window, low_res_y_pixel_expanded)
    # window 是 (C, 3, 3)，low_res_y_pixel_expanded 也是 (C, 3, 3)
    # cat 之后是 (2*C, 3, 3)
    
    if len(context_features) == 2: # 包含低分辨率像素和同分辨率邻居
        # 将低分辨率像素扩展到 3x3 形状，然后与窗口拼接
        low_res_y_pixel_expanded = low_res_y_pixel.expand(-1, context_window_size, context_window_size)
        context_tensor = torch.cat((window, low_res_y_pixel_expanded), dim=0)
    else: # 仅同分辨率邻居 (如处理 x(3) 的上下文，虽然论文说 x(3) 明文编码)
        context_tensor = window
        
    return context_tensor


class SReCModel(nn.Module):
    """
    SReC模型，包含多个用于不同分辨率级别的像素预测CNN。
    """
    def __init__(self, K=10, num_channels=3):
        super(SReCModel, self).__init__()
        self.K = K
        self.num_channels = num_channels
        
        # 修正3: 确保 PixelPredictorCNN 的 in_channels 与 get_pixel_context 的输出匹配
        # get_pixel_context 会返回 (2 * num_channels, 3, 3) 的上下文
        corrected_in_channels = 2 * num_channels 
        
        self.predictor_level0 = PixelPredictorCNN(corrected_in_channels, K)
        self.predictor_level1 = PixelPredictorCNN(corrected_in_channels, K)
        self.predictor_level2 = PixelPredictorCNN(corrected_in_channels, K)
        
        self.predictors = {
            0: self.predictor_level0,
            1: self.predictor_level1,
            2: self.predictor_level2
        }

    def forward(self, image_tensor):
        """
        前向传播，用于训练。计算给定图像的负对数似然损失。
        image_tensor: 原始图像 (C, H, W)
        返回: 总的 NLL 损失
        """
        x_levels, y_levels = generate_multi_resolution_images(image_tensor)
        
        total_nll_loss = 0.0
        
        # 遍历需要预测的级别 (0, 1, 2)
        for l in range(3):
            current_x = x_levels[l] # 目标图像级别
            C, H, W = current_x.shape
            
            # 遍历每个像素
            for i in range(H):
                for j in range(W):
                    # 获取像素上下文
                    context = get_pixel_context(x_levels, y_levels, l, i, j)
                    
                    # 确保上下文形状适合 CNN (batch_size, channels, H_context, W_context)
                    context = context.unsqueeze(0) # 添加 batch 维度 (1, C_effective, 3, 3)
                    
                    # 获取目标像素值
                    target_pixel_value = current_x[:, i, j].unsqueeze(0) # (1, C)
                    
                    # 通过 CNN 预测分布参数
                    weights_logits, means, scales = self.predictors[l](context)
                    # 修正4: weights_logits, means, scales 现在形状为 (1, K, 1, 1)
                    # 可以直接传递给 mixture_log_prob 和 mixture_entropy
                    
                    channel_nlls = []
                    for c_idx in range(self.num_channels):
                        log_prob = mixture_log_prob(
                            target_pixel_value[:, c_idx].unsqueeze(-1).unsqueeze(-1), # (1, 1, 1)
                            weights_logits, # (1, K, 1, 1)
                            means,          # (1, K, 1, 1)
                            scales          # (1, K, 1, 1)
                        )
                        channel_nlls.append(-log_prob.squeeze()) # 负对数似然
                    
                    total_nll_loss += torch.stack(channel_nlls).sum() # 累加所有通道的 NLL

        return total_nll_loss

# --- 训练脚本 ---

def train_srec_model(dataset_path, model_save_path, num_epochs=1, batch_size=1, learning_rate=1e-3):
    """
    训练 SReC 模型。
    dataset_path: 真实图像数据集的路径 (例如 'path/to/real_images_folder')
    model_save_path: 训练好的模型保存路径
    num_epochs: 训练轮数
    batch_size: 批处理大小 (SReC 像素级处理，这里简化为1)
    learning_rate: 学习率
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # 缩小图像尺寸以加速演示
        transforms.ToTensor(),       # 转换为张量，范围 [0, 1]
        transforms.Lambda(lambda x: x * 255) # 转换为 [0, 255]
    ])

    # 加载真实图像数据集
    try:
        dataset = ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"成功加载数据集，包含 {len(dataset)} 张图像。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保 'dataset_path' 指向一个包含图像的文件夹，例如 'data/real_images/'")
        print("并确保该文件夹下有子文件夹（例如 'class1', 'class2'），图像在子文件夹中。")
        print("或者，您可以手动创建 ImageFolder 结构或调整加载方式。")
        return

    # 初始化模型、优化器
    model = SReCModel(num_channels=3) # 假设 RGB 图像
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型将在 {device} 上运行。")

    model.train() # 设置为训练模式
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device) # 将图像移动到设备
            
            # 修正5: 添加 NaN/Inf 检查，确保输入数据没有问题
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"警告: 批次 {batch_idx} 图像包含 NaN 或 Inf 值，跳过此批次。")
                continue # 跳过当前批次

            optimizer.zero_grad()
            
            # SReC 的前向传播和损失计算是像素级的
            # 这里的 model.forward 已经包含了多分辨率处理和 NLL 累加
            loss = model(images.squeeze(0)) # 假设 batch_size=1，处理单张图像
            
            # 修正6: 检查损失是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch+1}, Batch {batch_idx+1}, 损失为 NaN/Inf。跳过此批次。")
                # 可以选择在这里调整学习率或采取其他恢复策略
                optimizer.zero_grad() # 清除可能已损坏的梯度
                continue

            loss.backward()
            # 修正7: 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm 可以调整
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), model_save_path)
    print(f"训练完成，模型已保存到 {model_save_path}")

# --- 运行训练 ---
if __name__ == '__main__':
    # 示例用法：
    # 1. 准备一个真实图像数据集文件夹，例如：
    #    data/
    #    └── real_images/
    #        └── class1/
    #            ├── img1.jpg
    #            └── img2.png
    #        └── class2/
    #            ├── img3.jpg
    #            └── img4.png
    # 2. 将 dataset_path 替换为您的真实图像数据集路径
    
    # 确保保存模型的目录存在
    os.makedirs('srec_models', exist_ok=True)

    # 请将 'path/to/your/real_images_dataset' 替换为您的真实图像数据集路径
    # 例如：'./data/real_images'
    real_images_dataset_path = 'train_dataset_150'
    srec_model_save_path = 'srec_models/srec_predictor_150_1.pth'

    print("--- 开始训练 SReC 模型 ---")
    # 修正8: 尝试更小的学习率，这是解决 NaN 的常见方法
    train_srec_model(real_images_dataset_path, srec_model_save_path, num_epochs=20, batch_size=1, learning_rate=1e-5) # 降低学习率
    print("--- SReC 模型训练结束 ---")

