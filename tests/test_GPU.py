import torch

# 检查是否有可用的CUDA设备
has_gpu = torch.cuda.is_available()
print(f"是否有可用GPU: {has_gpu}")

if has_gpu:
    # 查看GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")

    # 查看当前使用的GPU
    current_gpu = torch.cuda.current_device()
    print(f"当前使用的GPU索引: {current_gpu}")

    # 查看GPU名称
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"GPU名称: {gpu_name}")