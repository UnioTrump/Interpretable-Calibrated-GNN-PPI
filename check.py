import torch

checkpoint = torch.load('./checkpoints/best_model_loss.pt', map_location='cpu')
model_state = checkpoint['model_state_dict']

# 检查线性层的权重维度
if 'linear2.weight' in model_state:
    print(f"linear2.weight shape: {model_state['linear2.weight'].shape}")
    # 输出层的weight形状通常是 [output_dim, input_dim]
    out_channels = model_state['linear2.weight'].shape[0]
    print(f"Likely out_channels value: {out_channels}")

# 检查卷积层的输出维度
for key in model_state:
    if 'convs.1.bias' in key:  # 最后一层卷积的偏置通常反映输出通道数
        print(f"{key} shape: {model_state[key].shape}")