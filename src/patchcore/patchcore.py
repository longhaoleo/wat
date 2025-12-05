import logging
import os
import pickle
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


def to_nchw_if_vit(x: torch.Tensor, allow_dist_token: bool = True) -> torch.Tensor:
    """
    将 ViT 的序列特征 [B, N, C] 自动转换为 [B, C, H, W]；
    若已是 [B, C, H, W] 则原样返回。
    - 优先尝试直接开平方还原；
    - 不行则尝试去掉 1 个 CLS token；
    - 仍不行且允许 dist，则尝试去掉 2 个 token（CLS + dist）。
    若仍无法构成方形网格则抛错，明确提示 N 的因数分解。

    参数:
        x: torch.Tensor
        allow_dist_token: 是否允许尝试丢弃 dist token（如 DeiT）

    返回:
        torch.Tensor: [B, C, H, W]
    """
    if x.ndim == 4:
        return x  # CNN 分支，直接返回
    elif x.ndim == 3:
        B, N, C = x.shape
    else:
        raise ValueError(f"Expect 3D [B,N,C] or 4D [B,C,H,W], got {x.shape}")

    def try_square(tokens: torch.Tensor) -> torch.Tensor:
        B_, N_, C_ = tokens.shape
        H = int(N_ ** 0.5)
        if H * H == N_:
            return tokens.transpose(1, 2).reshape(B_, C_, H, H)
        return None

    # 1) 直接尝试不开窗（假设已去 CLS）
    out = try_square(x)
    if out is not None:
        return out

    # 2) 尝试去掉 1 个 token（CLS）
    if N > 1:
        out = try_square(x[:, 1:, :])
        if out is not None:
            return out

    # 3) （可选）尝试去掉 2 个 token（CLS + dist）
    if allow_dist_token and N > 2:
        out = try_square(x[:, 2:, :])
        if out is not None:
            return out

    # 4) 仍失败，友好报错
    #   给出 N 的因数，帮你定位为啥不是平方数
    factors = [d for d in range(2, min(N, 1024) + 1) if N % d == 0]
    raise ValueError(
        f"Cannot restore ViT grid from tokens: got N={N}, C={C}. "
        f"Tried removing CLS/dist but still not square. "
        f"Divisors of N (<=1024) = {factors}. "
        f"Hint: ensure CLS removed; if your model uses unusual tokens, adjust removal logic."
    )


class PatchCore(torch.nn.Module):
    """
    PatchCore异常检测类，基于图像补丁级别的特征进行异常检测。

    该类包含了训练和推理的实现，用于训练和评估PatchCore模型。
    """

    def __init__(self, device):
        """
        初始化PatchCore异常检测模型

        参数:
            device: [torch.device] 计算设备，通常为"cpu"或"cuda"。
        """
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        """
        加载 PatchCore 模型及其配置

        参数:
            backbone: [torch.nn.Module] 用作特征提取的神经网络骨干，通常是一个预训练的深度神经网络模型，如 ResNet。
            layers_to_extract_from: [list of str] 用于提取特征的层列表。每个元素是网络层的名称，表示从哪些层提取特征。
            device: [torch.device] 计算设备，指定模型在 CPU 或 GPU 上运行。
            input_shape: [tuple] 输入图像的形状，通常为 `(batch_size, channels, height, width)`。
            pretrain_embed_dimension: [int] 预训练嵌入的维度，指定从预训练网络中获取的特征维度。
            target_embed_dimension: [int] 目标嵌入的维度，指定训练后模型的特征空间维度。
            patchsize: [int] 补丁大小，表示切分图像时每个补丁的尺寸（默认为3）。
            patchstride: [int] 补丁步幅，表示切分图像时相邻补丁之间的步幅（默认为1）。
            anomaly_score_num_nn: [int] 用于异常评分的最近邻数量。控制在计算异常评分时，考虑多少个最近邻。
            featuresampler: [patchcore.sampler] 采样器，用于对特征进行采样处理。默认使用 `IdentitySampler`，表示不进行任何采样。
            nn_method: [patchcore.common.FaissNN] 最近邻计算方法，默认使用 `FaissNN`，基于 FAISS 库来加速最近邻搜索。
        """
        # 将传入的骨干网络模型（backbone）转移到指定的计算设备（CPU 或 GPU）
        self.backbone = backbone.to(device)

        # 将需要提取特征的层（`layers_to_extract_from`）保存为类的成员变量
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        # 创建 PatchMaker 实例，用于图像分块（patching）
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        # 使用 `torch.nn.ModuleDict` 来保存模块
        self.forward_modules = torch.nn.ModuleDict({})

        # 创建特征提取器：NetworkFeatureAggregator，使用指定的 backbone 和提取的层
        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        # 获取特征提取器输出的特征维度，输入的形状 `input_shape` 会传递给它
        # 计算每个抽取层的通道数（特征维）
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # 自动推断 pretrain_embed_dimension：当传入值为 None/0/负数时，
        # 取所选层通道数的最大值，作为预处理输出维度（自适应池化目标）
        auto_pre_dim = max(int(d) for d in feature_dimensions) if len(feature_dimensions) else target_embed_dimension
        pre_dim = int(pretrain_embed_dimension) if (pretrain_embed_dimension and pretrain_embed_dimension > 0) else auto_pre_dim

        # 配置预处理层：将每层展平后自适应池化到 pre_dim，再堆叠供后续聚合
        preprocessing = patchcore.common.Preprocessing(feature_dimensions, pre_dim)
        self.forward_modules["preprocessing"] = preprocessing

        # 配置嵌入层：目标嵌入维度用于训练后的特征空间转换
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(target_dim=target_embed_dimension)

        # 将嵌入层（`preadapt_aggregator`）移动到指定的计算设备
        _ = preadapt_aggregator.to(self.device)

        # 将 `preadapt_aggregator` 添加到 `forward_modules` 中
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # 配置最近邻异常评分器：`NearestNeighbourScorer`，使用指定的最近邻方法和数量
        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        # 配置异常分割器：`RescaleSegmentor`，用于图像分割，将模型预测结果映射到适当的大小
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        # 将采样器赋值给类的成员变量，默认使用 `IdentitySampler`（不进行特征采样）
        self.featuresampler = featuresampler

    def embed(self, data):
        """
        对输入数据计算 PatchCore 的特征嵌入。
        支持两种使用方式：
        1. 传入 DataLoader：遍历所有 batch，对每个 batch 调用 `_embed`；
        2. 传入单个 / 一批图像张量：直接调用 `_embed`。
        参数:
            data:
                - torch.utils.data.DataLoader：其中每个元素通常是图像张量，
                  或包含键 "image" 的字典；
                - 或者是可以直接送入模型的图像张量（形状一般为 [B, C, H, W]）。
        返回:
            features:
                - 若 data 为 DataLoader：返回一个列表 list[... ]，
                  每个元素是对应 batch 的嵌入特征（`_embed` 的返回值）；
                - 若 data 为单个 / 一批图像张量：返回一次 `_embed` 的结果。
        """
        # 如果传入的是 DataLoader，则逐 batch 提取特征
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                # 兼容 dataset 返回 dict 的情况（常见于包含多种字段的自定义 Dataset）
                if isinstance(image, dict):
                    image = image["image"]
                # 推理阶段关闭梯度，节省显存与加速
                with torch.no_grad():
                    # 转为 float32，并移动到当前模型所在的设备（CPU / GPU）
                    input_image = image.to(torch.float).to(self.device)
                    # 调用内部真正做特征提取与补丁处理的函数
                    features.append(self._embed(input_image))
            # 返回每个 batch 对应的嵌入特征列表
            return features

        # 若不是 DataLoader，则认为已经是一个可以直接送入 `_embed` 的批次
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """
        将输入图像批次映射为 PatchCore 所需的“补丁级特征嵌入”。
        功能概述：
            1. 用 backbone + 中间层钩子提取多层特征图；
            2. 对每层特征图进行补丁划分（patchify），得到每个空间位置对应的局部特征；
            3. 将不同层的补丁网格在空间上对齐到同一参考网格大小 (Gh_ref, Gw_ref)；
            4. 对每层的补丁特征做维度规整与跨层聚合，得到最终的 patch-level 向量表示。

        返回:
            长度 = 使用的层数
            若 provide_patch_shapes=False:
                features: list[np.ndarray 或 torch.Tensor]，
                        每个元素形状为 [B * (Gh * Gw), D, p, p] 经过预处理与聚合后，
                        实际将被整形成 [B * (Gh * Gw), T]（T是聚合目标维度）。
            若 provide_patch_shapes=True:
                (features, patch_shapes)
                features 同上；patch_shapes: list[Tuple[int,int]]，与 self.layers_to_extract_from 对齐，
                记录每个层被 patchify 后的网格大小 (Gh, Gw)（未对齐前的原生网格）。
        """

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # 冻结特征提取模块（backbone + 中间层钩子），并在 no_grad 下提取所有层输出
        # self.forward_modules["feature_aggregator"].eval()
        # 等于上面这个
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            # features_raw 是一个 dict: {layer_name: tensor[B, C_l, H_l, W_l]}
            features_raw = self.forward_modules["feature_aggregator"](images)

        # 仅保留我们配置要抽取的层（顺序与 self.layers_to_extract_from 一致）
        features = [features_raw[layer] for layer in self.layers_to_extract_from]
        # vit补丁：统一把 ViT [B,N,C] 变成 [B,C,H,W]；CNN 会原样返回
        features = [to_nchw_if_vit(x) for x in features]

        # features：[B, C_i, H_i, W_i]，来自不同层，变为 [ (unfolded_features, number_of_total_patches) ...]
        # 在特征图上做滑窗(kernel=patchsize, stride=patchstride),把每个局部 p × p 区域抽出来形成一个 patch。
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        # unfolded_features: Tensor： [B, Gh_i * Gw_i, C_i, p, p]
        # number_of_total_patches ：(Gh_i, Gw_i)
        patch_shapes = [x[1] for x in features]   
        features     = [x[0] for x in features]

        # 选择第一个层的补丁网格作为“参考网格尺寸”，后续将其它层双线性插值到这个网格
        # 这样做的原因：多层特征分辨率不同（例如 layer2 比 layer3 更高），需要对齐到统一的 (Gh_ref, Gw_ref)
        ref_num_patches = patch_shapes[0]          # (Gh_ref, Gw_ref)

        # 目标：把除参考层外的其它层的patch都整理成形状 [B * Gh_ref * Gw_ref, C, p, p]，
        # 使得空间位置一一对齐，只有一层则不执行
        # len(features) == len(self.layers_to_extract_from)
        for i in range(1, len(features)):
            _features = features[i]                # [B, Gh_i * Gw_i, C_i, p, p]
            patch_dims = patch_shapes[i]          # (Gh_i, Gw_i)

            # 恢复形状方便做空间插值:
            #     [B, Gh_i * Gw_i, C, p, p] -> [B, Gh_i, Gw_i, C, p, p]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )            

            # 置换维度，把(C, p, p)移到前面，方便后续合并与插值：
            _features = _features.permute(0, -3, -2, -1, 1, 2)  # [B, C, p, p, Gh_i, Gw_i]
            perm_base_shape = _features.shape                   # 记录以便还原

            # 把 (Gh_i, Gw_i) 展平到 batch 维上，准备对“空间网格”做双线性插值:
            # [B, C, p, p, Gh_i, Gw_i] -> [B*C*p*p, Gh_i, Gw_i]
            # -1 这个维度的大小让框架自动推断，保留最后两个维度用于插值
            _features = _features.reshape(-1, *_features.shape[-2:]) 

            # 对(Gh_i, Gw_i) 维度进行双线性插值到参考 (Gh_ref, Gw_ref)
            # 插入通道维，匹配 F.interpolate 的输入格式 [N, C, H, W]
            # 把这个层的 patch 网格从自己的大小拉伸/压缩到参考层的大小
            _features = F.interpolate(
                _features.unsqueeze(1),                    # [B*C*p*p, 1, Gh_i, Gw_i]
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)                                   # [B*C*p*p, Gh_ref, Gw_ref]

            # 还原回 [B, C, p, p, Gh_ref, Gw_ref]
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )

            # 再置换回 [B, Gh_ref, Gw_ref, C, p, p]
            _features = _features.permute(0, -2, -1, 1, 2, 3)

            # 再把数量堆叠起来 [B * Gh_ref * Gw_ref, C, p, p]
            _features = _features.reshape(-1, *_features.shape[-3:])
            features[i] = _features

        # 参考层也要展平 patch 维
        features[0] = features[0].reshape(-1, *features[0].shape[-3:])

        # 特征预处理与聚合
        # preprocessing: 不同层通道数 C_i 不同，将不同层的 (Ci*p*p) 通道映射/压缩到统一维度 pretrain_embed_dimension，
        # 并堆叠成 [B*Gh_ref*Gw_ref, #layers, pretrain_embed_dimension]
        # preadapt_aggregator: 自适应平均池化到 target_embed_dimension，
        # 并将层维聚合，得到最终 [B*Gh_ref*Gw_ref, target_embed_dimension]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        # 返回（可选 detach+CPU），以及是否返回每层的原生补丁网格尺寸
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """
        使用给定的训练数据构建 记忆库。
        
        参数:
            training_data :
                通常为 torch.utils.data.DataLoader，迭代输出训练图像。
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """
        计算训练数据的特征并填充“记忆库”。

        功能：
            - 将输入数据逐批送入模型，调用 `_embed` 得到补丁级嵌入特征；
            - 将所有 batch 的特征在样本维度上拼接；
            - 根据采样器 `self.featuresampler`（如随机采样、IdentitySampler 等）做可选降采样；
            - 调用 `self.anomaly_scorer.fit(...)`，用这些特征构建近邻搜索索引，
            作为后续异常检测阶段的“支持特征 / 记忆库”。

        参数:
            input_data :
                一般为 torch.utils.data.DataLoader，迭代输出训练图像（通常是正常样本），
                其元素可以是张量或包含键 "image" 的字典。
        """
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing features of memorybank ...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        # 拼接所有 batch 的特征，features.shape == [N_total, D]
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        # 对特征进行“建索引 / 建记忆库”
        # 把这个矩阵装进一个 list 传进去，接口设计支持多组特征，这里只用一组
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """
        对一个完整的 dataloader 执行推理，返回：
            scores, masks, labels_gt, masks_gt

        要求 dataloader 的每个 batch 是一个 dict，至少包含：
            - "image": Tensor[B, C, H, W]
            - "is_ai": Tensor[B] 或标量（可为 None）
            - "is_anomaly": Tensor[B] 或标量（可为 None）
        """
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        paths = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                # 这里 batch 一定是 dict
                images = batch.get("image")
                batch_size = images.shape[0]
                batch_paths = batch.get("image_path")
                label_tensor = batch.get("is_ai")

                if label_tensor is not None:
                    label_tensor = (
                        label_tensor.reshape(-1)
                        if isinstance(label_tensor, torch.Tensor)
                        else torch.as_tensor(label_tensor).reshape(-1)
                    )
                    labels_gt.extend(label_tensor.detach().cpu().numpy().tolist())
                else:
                    # 没有标签字段时直接报错，提示 dataloader 配置问题
                    raise ValueError(
                        "Batch 中未找到标签字段 'is_ai'，请检查 dataloader 的返回字典。")

                batch_scores, batch_masks = self._predict(images)
                scores.extend(batch_scores)
                masks.extend(batch_masks)
                paths.extend(batch_paths)

        return (scores, masks, labels_gt, paths)

    def _predict(self, images):
        """
        对一批图像进行异常检测推理，输出：
            - 每张图像的异常得分（score）
            - 对应的异常掩码/热力图（mask）

        整体流程：
            1) 将输入图像送入特征提取网络（feature_aggregator）
            得到每个补丁（patch）的特征嵌入。
            2) 使用 anomaly_scorer（通常是 KNN / FaissNN）计算每个补丁的异常得分。
            3) 将补丁得分重新映射为整张图的得分分布（mask）。
            4) 对每张图计算单一异常分数（平均或最大值）。
            5) 返回所有图像的得分与掩码。

        参数:
            images : torch.Tensor
                输入图像张量，形状 [B, C, H, W]，
                B 为 batch 大小，C 通道数（通常3），H/W 图像尺寸。

        返回:
            image_scores : list[float]
                每张图像的异常分数（越高表示越异常）。
            masks : list[np.ndarray]
                每张图像的异常热力图，用于可视化异常区域。
        """

        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        # 获取 batch 大小（后面 reshape、聚合时要用）
        batchsize = images.shape[0]

        with torch.no_grad():
            # 提取图像的 patch 特征嵌入
            # features: np.ndarray 或 Tensor，形状 [B*Gh*Gw, T]（T是target_embed_dimension）
            # patch_shapes: list[Tuple[int, int]]，长度 = 使用的层数
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)

            # 将特征转换为 numpy 数组，以便交给 FAISS/KNN 模块
            features = np.asarray(features)

            # 使用记忆库 (memory bank) 中的特征进行最近邻搜索，
            # 获取每个补丁（patch）的异常得分
            # 返回一个二维数组：每个补丁一个得分
            # predict([features]) -> [ [scores], ... ] [B*Gh*Gw]
            patch_scores = self.anomaly_scorer.predict([features])[0]

            # 将线性补丁得分重新组织为原图的 patch 网格
            # unpatch_scores: [B*Gh*Gw] -> [B, Gh*Gw]
            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )

            # 1.聚合patch得分为图像分数
            # self.patch_maker.score() 默认可选平均、最大或其他策略
            image_scores = self.patch_maker.score(image_scores,reduction='mean')

            # 2.生成patch得分热力图    
            # scales 即 patch 网格的行列数 (Gh, Gw)
            scales = patch_shapes[0]

            # 恢复为 [B, Gh_ref, Gw_ref]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            # anomaly_segmentor 将 patch-level mask 上采样回原图尺寸
            segmentation = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], segmentation


    @staticmethod
    def _params_file(filepath, prepend=""):
        """
        构造保存 PatchCore 配置参数的文件路径。
        参数:
            filepath : str
                保存目录（通常是某个实验 / 模型输出目录）。
            prepend : str, 可选
                文件名前缀，用于区分不同配置版本。
        返回:
            str: 完整的参数文件路径，例如：
                 "<filepath>/<prepend>patchcore_params.pkl"
        """
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        """
        将当前 PatchCore 模型的“记忆库 + 配置参数”保存到指定目录。

        内容包括两部分：
            1. anomaly_scorer 的索引数据（记忆库特征、FAISS 索引等）；
            2. PatchCore 本身的关键超参数（backbone、层名、patch 大小等），
               方便之后在同样配置下恢复模型。
        参数:
            save_path : str
                目标保存目录。
            prepend : str, 可选
                文件名前缀，用于在同一目录下区分多组 PatchCore 配置。
        """
        LOGGER.info("Saving PatchCore data.")

        # 1) 保存异常评分器（KNN / FAISS）的内部索引 / 特征
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )

        # 2) 组织 PatchCore 的关键配置参数，便于后续 load 时重建结构
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        # 写入一个 pickle 文件，保存上述配置字典
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        """
        从指定目录恢复并初始化一个 PatchCore 实例。

        恢复内容包括：
            1. PatchCore 的结构与超参数配置（backbone 名称、提取层、patch 大小等）；
            2. anomaly_scorer 的近邻索引 / 记忆库特征（与保存时配套）。

        参数:
            load_path : str
                保存模型的目录（与 save_to_path 使用的目录一致）。
            device : torch.device
                当前运行设备（如 torch.device("cuda") 或 "cpu"）。
            nn_method : patchcore.common.FaissNN 或其子类
                用于最近邻搜索的实现，默认使用 FaissNN(on_gpu=False, num_workers=4)。
            prepend : str, 可选
                文件名前缀，用于从特定前缀版本中加载（与保存时的 prepend 对应）。
        """
        LOGGER.info("Loading and initializing PatchCore.")

        # 1) 读取保存的 PatchCore 参数字典（结构配置）
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)

        # 2) 根据保存的 backbone 名称重新构建 backbone 模型
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]

        # 3) 调用 PatchCore.load(...)，用读取到的参数在当前实例上完成初始化
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        # 4) 恢复 anomaly_scorer 的索引 / 特征记忆库
        self.anomaly_scorer.load(load_path, prepend)


# 图像/特征的切块与还原相关工具
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        """
        参数:
            patchsize: 每个 patch 的边长 (kernel_size)
            stride:    滑动窗口步长；若为 None，需在外部保证传入时已设置
        """
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """
        将特征图按滑窗方式切成 patch。
        输入:
            features: Tensor，形状通常为 [B, C, H, W]
                      如果是 [N, D]（二维），会在后面补维成 [N, 1, 1, D] 以便 Unfold
        返回:
            若 return_spatial_info=False:
                unfolded_features: Tensor，形状 [B, Np, C, patch, patch]
                    其中 Np = patch 网格总数 = Gh * Gw
            若 return_spatial_info=True:
                (unfolded_features, number_of_total_patches)
                其中 number_of_total_patches: [Gh, Gw]，每个空间维度上的 patch 数
        """
        # 计算 patchsize 的一半作为 padding，保证边缘可切到完整 patch
        padding = int((self.patchsize - 1) / 2)

        # 若传入是二维向量 (N, D)，补成 4D，以便 Unfold 工作:
        # (N, D) -> (N, 1, 1, D)（把 D 当成“宽度”维，用 1×D 的滑窗切）
        if features.dim() == 2:
            features = features.unsqueeze(1).unsqueeze(2)

        # 使用 Unfold 实现滑窗切块
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1, # 膨胀系数，这里为 1 表示不跳点，连续像素。若 >1，相当于在窗口内部做“空洞卷积”
        )
        # Unfold 输入形状：[B, C, H, W]
        # Unfold 输出形状: [B, C*patchsize*patchsize, Np]
        # C*patchsize*patchsize：每个卷积块都把局部patch展开为一个向量，Np 是 每张图的 patch 数量
        unfolded_features = unfolder(features)

        # 计算每个空间维度上的 patch 数（Gh、Gw）
        # 公式来自 Unfold 的输出尺寸计算:
        # n_patches = floor((s + 2*padding - dilation*(patchsize-1) - 1)/stride + 1)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))  # [Gh, Gw]

        # 将 [B, C*patch*patch, Np] -> [B, C, patch, patch, Np]
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        ) # features 是 传进 patchify 的原始特征图
        # 再排列到 [B, Np, C, patch, patch]，方便后续按 patch 维度处理
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """
        将按 patch ，拆出 batch 维度。
        输入:
            x: 形状 [B*Gh*Gw]
            batchsize: 原先的 B
        输出:
            形状 [B, Gh*Gw]
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x, reduction: str = "mean"):
        """
        对输入张量/数组在其所有剩余维度上聚合，得到“每个样本的单一分数”。
        参数:
            reduction: "max"取全局最大值；"mean" 取全局均值。
        """
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)

        if reduction not in ("max", "mean"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        if x.ndim <= 1:
            out = x
        else:
            dims = tuple(range(1, x.ndim))
            out = torch.amax(x, dim=dims) if reduction == "max" else torch.mean(x, dim=dims)

        if was_numpy:
            return out.numpy()
        return out
