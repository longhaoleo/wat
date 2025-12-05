from pickle import FALSE
import torch
import torchvision
from patchcore.utils import visualize_vit_feature,visualize_conv_feature
from patchcore.datasets.tiny_genimage import Dataset
import os
from patchcore.backbones import load

def extract_conv_layers(model):
    """
    获取模型中与图像特征相关的卷积层（跳过分类头部分）

    参数：
    model (torch.nn.Module): 预训练的CNN模型

    返回：
    layers (dict): 包含特征提取部分卷积层的字典，键为层路径，值为卷积层对象
    """
    conv_layers = {}

    def _recursive_extract(module, parent_name=""):
        # 遍历当前模块的所有子模块
        for name, submodule in module.named_children():
            # 跳过分类头相关模块（如classifier、fc、head等）
            if name in ['classifier', 'fc', 'head', 'predictor']:
                continue  # 直接跳过分类部分，不递归处理
            
            # 构建当前子模块的完整路径
            current_name = f"{parent_name}.{name}" if parent_name else name
            
            # 如果是卷积层，添加到字典
            if isinstance(submodule, torch.nn.Conv2d):
                conv_layers[current_name] = submodule
            
            # 递归处理子模块（仅处理特征提取相关部分）
            _recursive_extract(submodule, current_name)

    # 从模型根节点开始递归提取
    _recursive_extract(model)
    return conv_layers

# 提取所有 Transformer 层
def extract_vit_layers(model):
    """
    遍历 ViT 模型，提取其中的所有 Transformer Layer（包含注意力层、MLP 层等）。
    """
    vit_layers = {}

    # 遍历模型的每一层
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.ModuleList):
            # 如果是 Transformer Block（ModuleList 包含多个子模块），则遍历其中的每一层
            for sub_name, sub_layer in layer.named_children():
                vit_layers[f"{name}.{sub_name}"] = sub_layer
        else:
            # 如果是其他层（例如 PatchEmbedding），直接添加
            vit_layers[name] = layer

    return vit_layers


def visualize(model, layers, dataloader, output_dir,VIT=False):
    # 遍历 DataLoader 中的每个批次，进行特征图可视化
    for batch_idx, input_image in enumerate(dataloader):
        print(f"Visualizing feature maps for batch {batch_idx + 1}")
        if VIT:
            visualize_vit_feature(model, layers, input_image['image'], output_dir=output_dir, batch_idx=batch_idx)
        else:
            visualize_conv_feature(model, layers, input_image['image'], output_dir=output_dir, batch_idx=batch_idx)
        
        # 假设你只想查看一个批次的特征图
        if batch_idx == 0:
            break  # 仅处理第一个批次并退出循环

if __name__ == "__main__":
    # 创建一个简单的 DataLoader 来加载数据集
    data_path = os.path.expanduser("~/datasets/tovisiualization/sdv5")
    dataset = Dataset(
        source=data_path,
        bankname="ai",
        name='sdv5'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


    model_name = "bninception"
    # 加载预训练模型
    model =load(model_name)
    model.eval()  # 设置为评估模式

    # 保存特征图的目录
    output_dir = os.path.expanduser(f"~/dreamycore/visualization/{model_name}")
    # 选择多个感兴趣的卷积层
    # layers = {
    #     'layer1.0.conv1': model.layer1[0].conv1,  # 第一层的卷积
    #     'layer2.0.conv1': model.layer2[0].conv1,  # 第二层的卷积
    #     'layer3.0.conv1': model.layer3[0].conv1,  # 第三层的卷积
    #     'layer4.0.conv1': model.layer4[0].conv1   # 第四层的卷积
    # }

    VIT = False
    if VIT:
        layers = extract_vit_layers(model)
    else:
        layers = extract_conv_layers(model)

    print(f"网络结构：{layers.keys()}")

    visualize(model=model, layers=layers, dataloader=dataloader, output_dir=output_dir,VIT=VIT)