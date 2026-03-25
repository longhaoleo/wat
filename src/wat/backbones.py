"""
骨干网络加载工具。

职责：
1) 维护可用 backbone 名称到“构造表达式”的映射；
2) 兼容 timm / torchvision / open_clip 三类来源；
3) 对外暴露统一入口 `load(name)`，让上层不用关心底层框架差异。

对应 `PROJECT_DETAILED_COMMENTS.md` 第 7 节：
- `_build_clip_backbone(...)`：创建 CLIP 视觉分支；
- `load(...)`：统一路由到 timm / torchvision / open_clip。
"""

import timm  # noqa
import torchvision.models as models  # noqa
# import pretrainedmodels

# 推荐安装 open_clip_torch 用于 CLIP 骨干网络
import open_clip


_BACKBONES = {
    # 经典 CNN
    "alexnet": "models.alexnet(pretrained=True)",
    # 批归一化 + Inception 多尺度卷积
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    # 轻量级 CNN（移动端友好）
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    # DenseNet：跨层特征复用
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    # 多尺度结构
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

_CLIP_BACKBONES = {
    # 使用 open_clip 对应的名称，视觉骨干为 ViT-B/16 与 ViT-B/32
    # 仅返回 .visual 分支，方便与现有 CNN / ViT 一致地抽特征
    # 这里使用 open_clip 官方在 LAION-2B 数据集上训练的权重标签
    "clip_vit_b16": ("ViT-B-16", "openai"),
    "clip_vit_b32": ("ViT-B-32", "openai"),
}


def _build_clip_backbone(name: str):
    """
    构建 CLIP 视觉骨干（依赖 open_clip_torch）。

    返回 `model.visual`，这样上层可以像普通 backbone 一样注册中间层 hook。
    """
    if name not in _CLIP_BACKBONES:
        raise KeyError(f"Unknown CLIP backbone: {name}. Supported: {list(_CLIP_BACKBONES.keys())}")

    arch, pretrained_tag = _CLIP_BACKBONES[name]
    # 这里直接从 open_clip 创建模型，返回其中的视觉分支
    # open_clip 会根据 `pretrained_tag` 下载对应权重，如 "openai"、"laion2b_s32b_b79k" 等
    model = open_clip.create_model(arch, pretrained=pretrained_tag)
    return model.visual


def load(name: str):
    """
    加载指定名称的 backbone。

    支持的形式:
        - 标准名称: 'resnet50', 'vit_base', ...
        - CLIP 视觉骨干: 'clip_vit_b16', 'clip_vit_b32'
    """
    # 常规 timm/torchvision 骨干
    if name in _BACKBONES:
        model = eval(_BACKBONES[name])
    # CLIP 视觉骨干（open_clip）
    elif name in _CLIP_BACKBONES:
        model = _build_clip_backbone(name)
    else:
        raise KeyError(
            f"Unknown backbone: {name}. "
            f"Available: {list(_BACKBONES.keys()) + list(_CLIP_BACKBONES.keys())}"
        )

    return model
