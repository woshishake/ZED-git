import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_zed_features(csv_file_path):
    """
    读取ZED特征CSV文件，并绘制|D(2)|与Source的散点图。

    Args:
        csv_file_path (str): ZED特征CSV文件的路径。
    """
    if not os.path.exists(csv_file_path):
        print(f"错误: 未找到CSV文件 '{csv_file_path}'。请确保文件存在且路径正确。")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        print("请检查CSV文件是否损坏或格式是否正确。")
        sys.exit(1)

    # 确保所需列存在
    required_columns = ['Source', '|D(2)|']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少列 '{col}'。请检查CSV文件内容。")
            sys.exit(1)

    # 将 '|D(2)|' 列转换为数值类型，处理可能的非数值数据
    df['|D(2)|'] = pd.to_numeric(df['|D(2)|'], errors='coerce')
    # 删除转换失败（NaN）的行
    df.dropna(subset=['|D(2)|'], inplace=True)

    if df.empty:
        print("警告: 经过数据清洗后，没有可用于绘图的数据。")
        return

    # 设置matplotlib的中文显示（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

    plt.figure(figsize=(12, 7)) # 设置图表大小

    # 使用seaborn绘制散点图
    # x轴为'Source'，y轴为'|D(2)|'
    # hue='Source' 可以根据来源自动着色，增强区分度
    # jitter=True 可以让散点在x轴上稍微错开，避免重叠，更清晰地看到数据点分布
    sns.stripplot(x='Source', y='|D(2)|', data=df, jitter=0.2, hue='Source', palette='viridis', legend=False)

    plt.title('不同来源图片 |D(2)| 特征散点图') # 图表标题
    plt.xlabel('图片来源 (Source)') # x轴标签
    plt.ylabel('|D(2)| 特征值') # y轴标签
    plt.grid(True, linestyle='--', alpha=0.7) # 添加网格线
    plt.tight_layout() # 自动调整布局，防止标签重叠
    plt.show() # 显示图表

if __name__ == '__main__':
    # 示例用法：
    # 命令行运行时输入 python ./analyze_zed.py zed_features_output.csv
    # 或者直接在脚本中指定文件路径
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python ./analyze_zed.py <ZED特征CSV文件路径>")
        print("例如: python ./analyze_zed.py zed_features_output.csv")
        sys.exit(1)
    
    input_csv_file = sys.argv[1]
    plot_zed_features(input_csv_file)
