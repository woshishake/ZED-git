import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys # 导入 sys 模块用于处理命令行参数
import math
import csv # 导入 csv 模块用于写入 CSV 文件

# --- SReC 核心组件实现 (与之前保持一致) ---

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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU()
        )
        self.param_predictor = nn.Conv2d(32, K * 3, kernel_size=1)

    def forward(self, context):
        features = self.conv_layers(context)
        params = self.param_predictor(features)
        params_reshaped = params.view(params.size(0), self.K, 3, params.size(2), params.size(3))
        
        weights_logits = params_reshaped[:, :, 0, :, :] 
        means = params_reshaped[:, :, 1, :, :]         
        scales = torch.exp(params_reshaped[:, :, 2, :, :])
        scales = torch.clamp(scales, max=1e5) 

        return weights_logits, means, scales

def logistic_cdf(x, mu, s):
    """
    计算离散逻辑分布的CDF（累积分布函数）差值，用于表示像素值概率。
    """
    s = torch.clamp(s, min=1e-5)
    sigmoid = torch.sigmoid
    cdf_upper = sigmoid((x - mu + 0.5) / s)
    cdf_lower = sigmoid((x - mu - 0.5) / s)
    return cdf_upper - cdf_lower

def mixture_log_prob(target_pixel_value, weights_logits, means, scales):
    """
    计算给定目标像素值在混合逻辑分布下的对数概率。
    """
    batch_size, K, H, W = weights_logits.shape
    weights = torch.softmax(weights_logits, dim=1)
    target_pixel_value_expanded = target_pixel_value.unsqueeze(1).expand(-1, K, -1, -1)
    component_probs = logistic_cdf(target_pixel_value_expanded, means, scales)
    mixture_probs = torch.sum(weights * component_probs, dim=1)
    mixture_probs = torch.clamp(mixture_probs, min=1e-10)
    return torch.log(mixture_probs)

def mixture_entropy(weights_logits, means, scales, pixel_range=256):
    """
    计算混合逻辑分布的熵。
    """
    batch_size, K, H, W = weights_logits.shape
    weights = torch.softmax(weights_logits, dim=1)
    all_pixel_values = torch.arange(0, pixel_range, device=means.device, dtype=means.dtype)
    
    all_pixel_values_expanded = all_pixel_values.view(1, 1, pixel_range, 1, 1).expand(batch_size, K, -1, H, W)
    means_expanded = means.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    scales_expanded = scales.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    weights_expanded = weights.unsqueeze(2).expand(-1, -1, pixel_range, -1, -1)
    
    component_probs_all_k = logistic_cdf(all_pixel_values_expanded, means_expanded, scales_expanded)
    mixture_probs_all_k = torch.sum(weights_expanded * component_probs_all_k, dim=1)
    mixture_probs_all_k = torch.clamp(mixture_probs_all_k, min=1e-10)
    
    entropy_map = -torch.sum(mixture_probs_all_k * torch.log(mixture_probs_all_k), dim=1)
    return entropy_map

def _2x2_avg_pool(image_tensor):
    """
    对图像张量进行 2x2 平均池化。
    """
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    return pool(image_tensor.unsqueeze(0)).squeeze(0)

def _round_to_int(image_tensor):
    """
    将浮点图像张量四舍五入并转换为整数（0-255）。
    """
    return torch.round(image_tensor).clamp(0, 255).to(torch.float32)

def generate_multi_resolution_images(image_tensor):
    """
    生成图像的多分辨率版本 x(l) 和 y(l)。
    """
    x_levels = [image_tensor]
    y_levels = []
    current_x = image_tensor
    for l in range(3):
        y_next = _2x2_avg_pool(current_x)
        y_levels.append(y_next)
        x_next = _round_to_int(y_next)
        x_levels.append(x_next)
        current_x = x_next
    return x_levels, y_levels

def get_pixel_context(x_levels, y_levels, level, i, j, context_window_size=3):
    """
    获取像素 (i,j) 在指定级别 l 的上下文 X(l)i,j。
    """
    current_x = x_levels[level]
    C, H, W = current_x.shape

    low_res_y_pixel = torch.zeros((C, 1, 1), device=current_x.device, dtype=current_x.dtype)
    if level < 3: 
        low_res_h = y_levels[level].shape[1]
        low_res_w = y_levels[level].shape[2]
        
        target_i_low_res = i // 2
        target_j_low_res = j // 2
        
        if target_i_low_res < low_res_h and target_j_low_res < low_res_w:
            low_res_y_pixel = y_levels[level][:, target_i_low_res, target_j_low_res].unsqueeze(-1).unsqueeze(-1)

    pad_size = context_window_size // 2
    padded_x = torch.nn.functional.pad(current_x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    start_i = i
    end_i = i + context_window_size
    start_j = j
    end_j = j + context_window_size
    
    window = padded_x[:, start_i : end_i, start_j : end_j]

    if window.shape[1] != context_window_size or window.shape[2] != context_window_size:
        print(f"警告: 级别 {level} 像素 ({i},{j}) 提取的窗口形状为 {window.shape} 而非 ({C}, {context_window_size}, {context_window_size})")
        window = torch.nn.functional.interpolate(window.unsqueeze(0), size=(context_window_size, context_window_size), mode='bilinear', align_corners=False).squeeze(0)

    low_res_y_pixel_expanded = low_res_y_pixel.expand(-1, context_window_size, context_window_size)
    context_tensor = torch.cat((window, low_res_y_pixel_expanded), dim=0)
        
    return context_tensor

class SReCModel(nn.Module):
    """
    SReC模型，包含多个用于不同分辨率级别的像素预测CNN。
    这里仅用于加载已训练的模型权重。
    """
    def __init__(self, K=10, num_channels=3):
        super(SReCModel, self).__init__()
        self.K = K
        self.num_channels = num_channels
        
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
        raise NotImplementedError("SReCModel's forward is not implemented for direct use in feature extraction. Use SReCFeatureExtractor.")


class SReCFeatureExtractor:
    """
    基于训练好的 SReC 模型提取图像特征（NLL, H, D, Delta）。
    """
    def __init__(self, model_path, K=10, num_channels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SReCModel(K=K, num_channels=num_channels).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # 设置为评估模式
            print(f"成功加载模型权重来自: {model_path}")
        except FileNotFoundError:
            print(f"错误: 未找到模型文件 '{model_path}'。请先运行训练脚本并确保路径正确。")
            sys.exit(1)
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            sys.exit(1)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), # 确保与训练时尺寸一致
            transforms.ToTensor(),       # 转换为张量，范围 [0, 1]
            transforms.Lambda(lambda x: x * 255) # 转换为 [0, 255]
        ])
        self.num_channels = num_channels

    def extract_features(self, image_path):
        """
        从图像中提取 ZED 特征。
        image_path: 待检测图像的路径。
        返回: 包含 D(l), Delta01, Delta02 及其绝对值的字典。
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)
        except FileNotFoundError:
            print(f"错误: 未找到图像文件 '{image_path}'。")
            return None
        except Exception as e:
            print(f"加载或处理图像时发生错误: {e}")
            return None

        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print(f"警告: 输入图像 '{image_path}' 包含 NaN 或 Inf 值，无法处理。")
            return None

        x_levels, y_levels = generate_multi_resolution_images(image_tensor)
        
        avg_nlls = {}
        avg_hs = {}

        with torch.no_grad(): # 在特征提取时不计算梯度
            for l in range(3): # 级别 0, 1, 2
                current_x = x_levels[l]
                C, H, W = current_x.shape
                
                pixel_nlls = torch.zeros((H, W), device=self.device)
                pixel_hs = torch.zeros((H, W), device=self.device)

                for i in range(H):
                    for j in range(W):
                        context = get_pixel_context(x_levels, y_levels, l, i, j)
                        context = context.unsqueeze(0) # 添加 batch 维度
                        
                        target_pixel_value = current_x[:, i, j].unsqueeze(0) # (1, C)

                        weights_logits, means, scales = self.model.predictors[l](context)
                        
                        channel_nlls = []
                        channel_hs = []
                        for c_idx in range(self.num_channels):
                            log_prob = mixture_log_prob(
                                target_pixel_value[:, c_idx].unsqueeze(-1).unsqueeze(-1),
                                weights_logits,
                                means,
                                scales
                            )
                            channel_nlls.append(-log_prob.squeeze())
                            
                            entropy_val = mixture_entropy(
                                weights_logits,
                                means,
                                scales
                            )
                            channel_hs.append(entropy_val.squeeze())
                        
                        pixel_nlls[i, j] = torch.stack(channel_nlls).sum()
                        pixel_hs[i, j] = torch.stack(channel_hs).sum()

                avg_nlls[l] = pixel_nlls.mean().item()
                avg_hs[l] = pixel_hs.mean().item()

        # --- 计算决策统计量 ---
        D_features = {}
        Delta_features = {}

        for l in range(3):
            D_features[l] = avg_nlls[l] - avg_hs[l]
            
        Delta_features['01'] = D_features[0] - D_features[1]
        Delta_features['02'] = D_features[0] - D_features[2]

        results = {
            'D(0)': D_features[0],
            'D(1)': D_features[1],
            'D(2)': D_features[2],
            'Delta01': Delta_features['01'],
            'Delta02': Delta_features['02'],
            '|D(0)|': abs(D_features[0]),
            '|D(1)|': abs(D_features[1]),
            '|D(2)|': abs(D_features[2]),
            '|Delta01|': abs(Delta_features['01']),
            '|Delta02|': abs(Delta_features['02']),
        }
        
        return results

# --- 运行特征提取 ---
if __name__ == '__main__':
    # 示例用法：
    # 1. 确保已运行训练脚本并保存了模型 (srec_predictor.pth)
    # 2. 命令行运行时输入 python ./test.py /path/to/your/test_images_folder
    #    例如: python ./test.py test_dataset
    
    srec_model_path = 'srec_models/srec_predictor_150_1.pth'
    output_csv_path = 'zed_features_output_150.csv' # 输出 CSV 文件名

    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python ./test.py <测试图片文件夹路径>")
        print("例如: python ./test.py test_dataset")
        sys.exit(1)
    
    test_root_path = sys.argv[1] # 根测试文件夹路径

    print("--- 开始提取 ZED 特征 ---")
    feature_extractor = SReCFeatureExtractor(srec_model_path)
    
    if not os.path.isdir(test_root_path):
        print(f"错误: 根文件夹 '{test_root_path}' 不存在或不是一个有效的目录。")
        sys.exit(1)

    # 支持的图片文件扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    # 打开 CSV 文件准备写入
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Image Name', 'Source', 'D(0)', 'D(1)', 'D(2)', 
                      'Delta01', 'Delta02', '|D(0)|', '|D(1)|', '|D(2)|', 
                      '|Delta01|', '|Delta02|']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader() # 写入 CSV 表头

        # 遍历根文件夹及其所有子文件夹
        for dirpath, dirnames, filenames in os.walk(test_root_path):
            # 获取当前子文件夹的名称作为来源
            # 如果是根目录本身，来源可以设为 'root' 或空
            source = os.path.basename(dirpath) if dirpath != test_root_path else 'root'
            if not source: # 处理根目录是 '/' 或 'C:\' 的情况
                source = 'root'

            for filename in filenames:
                if filename.lower().endswith(image_extensions):
                    image_full_path = os.path.join(dirpath, filename)
                    print(f"\n--- 处理图片: {filename} (来源: {source}) ---")
                    
                    features = feature_extractor.extract_features(image_full_path)
                    
                    if features:
                        print("提取到的 ZED 特征:")
                        row_data = {'Image Name': filename, 'Source': source}
                        for key, value in features.items():
                            print(f"    {key}: {value:.4f}")
                            row_data[key] = f"{value:.4f}" # 格式化为字符串，保留4位小数
                        writer.writerow(row_data) # 写入一行数据到 CSV
                    else:
                        print(f"图片 '{filename}' (来源: {source}) 特征提取失败。")
                else:
                    print(f"跳过非图片文件: {filename} (在 {dirpath} 中)")

    print(f"\n--- ZED 特征提取结束。结果已保存到 '{output_csv_path}' ---")
