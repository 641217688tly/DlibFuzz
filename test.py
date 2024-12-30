import torch
import torch.nn.functional as F
input_tensor = torch.tensor([
    [
        [1, 2, 3, 4, 5, 6, 7, 8],    # 第一个样本的第一个通道
        [9, 10, 11, 12, 13, 14, 15, 16],  # 第一个样本的第二个通道
        [17, 18, 19, 20, 21, 22, 23, 24]  # 第一个样本的第三个通道
    ],
    [
        [25, 26, 27, 28, 29, 30, 31, 32],  # 第二个样本的第一个通道
        [33, 34, 35, 36, 37, 38, 39, 40],  # 第二个样本的第二个通道
        [41, 42, 43, 44, 45, 46, 47, 48]   # 第二个样本的第三个通道
    ]
], dtype=torch.float)
output_length = 4
output_tensor = F.adaptive_avg_pool1d(input_tensor, output_length)