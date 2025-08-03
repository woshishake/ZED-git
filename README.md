尝试复现 Zero-Shot Detection of AI-Generated Images 的工作，参考 https://grip-unina.github.io/ZED/ ，以及 https://github.com/grip-unina/ZED/ 。

数据集获取：
  下载RAISE数据集和synthbuster数据集 https://github.com/grip-unina/ClipBased-SyntheticImageDetection/tree/c76ef7f5e158c5aba9e55b8b94ab0079720d281e/data
  参考 https://deepwiki.com/grip-unina/ClipBased-SyntheticImageDetection/3.3-synthbuster-dataset

训练：从RAISE_1k选取真实图像进行训练，注意修改 train.py 中数据集路径，模型保存至 srec_models 。
  - python ./train.py 

测试：从synthbuster数据集选取AIGI图像测试，结果输出csv文件。
  - python ./test.py test_dataset

分析：读取csv文件，绘图
  - python ./analyze.py zed_features_output.csv
