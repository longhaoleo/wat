import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)

from PIL import Image
from torchvision import transforms as T



def plot_random_segmentations(
    image_paths: list,
    segmentations_a: list,
    segmentations_b: list,
    mask_paths: str = "-1",
    savefolder: str = "output_images",
    num_samples: int = 50,
    save_depth: int = 4,
    resize: int = None,
    imagesize: int = None,
    annotations: list | None = None,
):
    """
    随机选择若干张图像，显示它们的异常分割结果，并保存分割图像到指定文件夹。

    参数:
        image_folder (str): 存放图像的文件夹路径。
        mask_folder (str): 存放掩码的文件夹路径。
        segmentations (list): 生成的异常分割结果列表，格式为 np.ndarray 列表。
        anomaly_scores (list): 异常得分列表，按图像顺序排列。
        savefolder (str): 结果保存的文件夹路径，默认 "output_images"。
        num_samples (int): 随机选择的图像数量，默认选择 20 张图像。
        save_depth (int): 路径深度，决定保存时文件夹的层级结构，默认为 4。
        resize (int): 若提供，则先对原图做 Resize(size, bilinear) 与训练保持一致。
        imagesize (int): 若提供（且与 resize 同时给出），会继续做 CenterCrop(imagesize)。
    """


    # 随机选择 num_samples 张图像进行显示
    sample_indices = random.sample(range(len(image_paths)), num_samples)
    sample_image_paths = [image_paths[i] for i in sample_indices]
    sample_segmentations_a = [segmentations_a[i] for i in sample_indices]
    sample_segmentations_b = [segmentations_a[i] for i in sample_indices]
    sample_annotations = [annotations[i] for i in sample_indices] if annotations else [None] * len(sample_indices)

    # 创建保存文件夹
    os.makedirs(savefolder, exist_ok=True)

    preprocess = None
    if resize is not None and imagesize is not None:
        preprocess = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(imagesize),
            ]
        )

    # 执行可视化并保存分割图像
    for image_path, segmentation_a, segmentation_b, ann in tqdm.tqdm(
        zip(sample_image_paths, sample_segmentations_a, sample_segmentations_b, sample_annotations),
        total=num_samples,
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        # 打开图像并进行必要的转换
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if preprocess is not None:
            image = preprocess(image)
        image = np.array(image)

        # 保存路径处理
        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        
        # 绘制图像、掩码和分割结果
        f, axes = plt.subplots(1, 3) 
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis('off')
        # 可选：在图中绘制文本标注（如两个库的分数）
        if ann is not None:
            try:
                axes[0].text(
                    0.02, 0.98, ann,
                    fontsize=9, color='yellow', ha='left', va='top',
                    transform=axes[0].transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none')
                )
            except Exception:
                pass
        axes[1].imshow(segmentation_a)
        axes[1].set_title("Segmentation_ai")
        axes[1].axis('off')

        axes[2].imshow(segmentation_b)
        axes[2].set_title("Segmentation_nature")
        axes[2].axis('off')
        # 调整布局并保存
        f.set_size_inches(12, 4)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

    print(f"Saved {num_samples} segmentation images to: {savefolder}")


def visualize_conv_feature(model, layers, input_image, output_dir=None, batch_idx=None, max_display=64,layername="None"):
    """
    可视化多个卷积层的特征图（Feature Maps）

    参数：
    model (torch.nn.Module): 预训练的CNN模型
    layers (list): 一个包含多个卷积层的列表
    input_image (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width]
    output_dir (str): 保存图像的目录路径。如果为 None，默认不保存。
    batch_idx (int): 当前批次的索引，用于生成唯一的文件名。

    输出：保存所有特征图到文件（如果指定了 output_dir），否则显示特征图。
    """
    for num, (layer_name, layer) in enumerate(layers.items()):
        print(f"Visualizing feature map for num {num + 1}")
        
        # 存储每个特征图
        def hook_fn(module, input, output):
            feature_maps.append(output)

        feature_maps = []
        
        # 注册hook来获取卷积层输出
        hook = layer.register_forward_hook(hook_fn)
        
        # 模型前向传播
        model(input_image)
        
        # 移除hook
        hook.remove()
        
        # 创建一个足够大的图来容纳所有子图,不同通道的
        num_feature_maps = feature_maps[0].shape[1]  # 特征图通道数量
        display_count = min(num_feature_maps, max_display)  # 限制显示的最大特征图数量
        rows = (display_count // 8)   # 计算需要多少行显示
        fig, axes = plt.subplots(rows, 8, figsize=(16, 2 * rows))  # 8列显示
        axes = axes.flatten()
        
        # 绘制原图作为第一个子图
        ax = axes[0]
        ax.imshow(input_image[0].cpu().detach().numpy().transpose(1, 2, 0))  # 将输入图像转换为 [height, width, channels] 格式
        ax.axis('off')  # 关闭坐标轴
        ax.set_title("Input Image")  # 原图标题

        # 绘制每个特征图
        for i in range(display_count-1):
            ax = axes[i + 1]  # 从第二个位置开始绘制特征图
            ax.imshow(feature_maps[0][0, i].cpu().detach().numpy(), cmap='viridis')
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f"Feature Map {i + 1}")  # 每个特征图标题

        # 如果提供了输出目录，保存图片
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # 使用批次索引和层索引来创建唯一的文件名
            filename = f"{layer_name}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Saved feature map for batch {batch_idx}, layer {num + 1} to {filepath}")
        
        # 显示特征图
        # plt.show()
        plt.close(fig)


def visualize_vit_feature(model, layers, input_image, output_dir=None, batch_idx=None, max_display=64, layername="None"):
    """
    可视化多个 ViT 层的特征图（Feature Maps），支持自适应重塑。

    参数：
    model (torch.nn.Module): 预训练的 ViT 模型
    layers (dict): 一个包含多个层的字典
    input_image (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width]
    output_dir (str): 保存图像的目录路径。如果为 None，默认不保存。
    batch_idx (int): 当前批次的索引，用于生成唯一的文件名。
    max_display (int): 每层最多显示的特征图数量，默认显示前16个。
    layername (str): 层的名称，用于命名保存文件。

    输出：保存所有特征图到文件（如果指定了 output_dir），否则显示特征图。
    """
    for num, (layer_name, layer) in enumerate(layers.items()):
        print(f"Visualizing feature map for layer {layer_name}")

        # 存储每个特征图
        def hook_fn(module, input, output):
            feature_maps.append(output)

        feature_maps = []

        # 注册hook来获取ViT层输出
        hook = layer.register_forward_hook(hook_fn)

        # 模型前向传播
        model(input_image)

        # 移除hook
        hook.remove()

        # 获取特征图的形状
        feature_map = feature_maps[0][0].cpu().detach().numpy()  # 获取第一个批次的特征图

        # 打印特征图的形状，帮助调试
        print(f"Feature map shape: {feature_map.shape}")

        # 将特征图 reshape 为二维图像
        if feature_map.ndim == 1:  # 如果是 [embedding_dim,]
            # 计算合适的形状（假设它是正方形）
            side_length = int(np.sqrt(feature_map.shape[0]))  # 计算正方形的边长
            if side_length * side_length != feature_map.shape[0]:
                side_length += 1  # 调整为接近的合适形状
            feature_map_reshaped = feature_map.reshape(side_length, side_length)  # 重塑为 2D 图像
        elif feature_map.ndim == 2:  # 如果是 [embedding_dim, num_patches] 的形状
            num_patches, embedding_dim = feature_map.shape
            side_length = int(np.sqrt(num_patches))  # 假设它可以变为正方形
            feature_map_reshaped = feature_map.reshape(side_length, side_length, embedding_dim)  # 重塑为 2D 图像
        else:
            feature_map_reshaped = feature_map  # 如果已经是二维或适合的形状，可以直接使用

        # 创建一个足够大的图来容纳所有子图
        rows = (feature_map_reshaped.shape[0] // 8) + 1  # 动态计算行数
        cols = 8  # 设定列数
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
        axes = axes.flatten()

        # 绘制原图作为第一个子图
        ax = axes[0]
        ax.imshow(input_image[0].cpu().detach().numpy().transpose(1, 2, 0))  # 将输入图像转换为 [height, width, channels] 格式
        ax.axis('off')  # 关闭坐标轴
        ax.set_title("Input Image")  # 原图标题

        # 绘制每个特征图
        for i in range(min(feature_map_reshaped.shape[0], max_display)):
            ax = axes[i + 1]  # 从第二个位置开始绘制特征图
            ax.imshow(feature_map_reshaped[i], cmap='viridis')
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f"Feature Map {i + 1}")  # 每个特征图标题

        # 隐藏多余的子图
        for i in range(min(feature_map_reshaped.shape[0], max_display), len(axes)):
            axes[i].axis('off')

        # 如果提供了输出目录，保存图片
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{layername}_batch_{batch_idx}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Saved feature map for batch {batch_idx}, layer {layer_name} to {filepath}")

        # 显示特征图
        plt.show()

        # 关闭当前图形，释放内存
        plt.close(fig)

def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
