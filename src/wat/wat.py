import logging
import os
import pickle
from typing import List

import numpy as np
import torch
import tqdm

import wat
import wat.backbones
import wat.common
import wat.sampler

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



class WAT(torch.nn.Module):
    """
    WAT（Without Any Train）异常检测：
    - 仅依赖特征提取 + KNN 记忆库，无梯度训练；
    - 支持记录每张图对应的数据集名称，推理时可回溯最近邻来源。
    """

    def __init__(self, device):
        """
        初始化 WAT 异常检测模型（无梯度训练，仅建库）

        参数:
            device: [torch.device] 计算设备，通常为"cpu"或"cuda"。
        """
        super(WAT, self).__init__()
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
        featuresampler=wat.sampler.IdentitySampler(),
        nn_method=wat.common.BruteNN(),
        **kwargs,
    ):
        """
        加载 WAT 模型及其配置（仅建 KNN 记忆库，无监督训练）

        参数:
            backbone: [torch.nn.Module] 用作特征提取的神经网络骨干，通常是一个预训练的深度神经网络模型，如 ResNet。
            layers_to_extract_from: [list of str] 用于提取特征的层列表。每个元素是网络层的名称，表示从哪些层提取特征。
            device: [torch.device] 计算设备，指定模型在 CPU 或 GPU 上运行。
            input_shape: [tuple] 输入图像的形状，通常为 `(batch_size, channels, height, width)`。
            pretrain_embed_dimension: [int] 预训练嵌入的维度，指定从预训练网络中获取的特征维度。
            target_embed_dimension: [int] 目标嵌入的维度，指定训练后模型的特征空间维度。
            patchsize: [int] （已废弃，占位）
            patchstride: [int] （已废弃，占位）
            anomaly_score_num_nn: [int] 用于异常评分的最近邻数量。控制在计算异常评分时，考虑多少个最近邻。
            featuresampler: [wat.sampler] 采样器，用于对特征进行采样处理。默认使用 `IdentitySampler`，表示不进行任何采样。
            nn_method: 最近邻计算方法，可选 `FaissNN`（如已安装）或默认 `BruteNN`。
        """
        # 将传入的骨干网络模型（backbone）转移到指定的计算设备（CPU 或 GPU）
        self.backbone = backbone.to(device)

        # 将需要提取特征的层（`layers_to_extract_from`）保存为类的成员变量
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        # 使用 `torch.nn.ModuleDict` 来保存模块
        self.forward_modules = torch.nn.ModuleDict({})

        # 创建特征提取器：NetworkFeatureAggregator，使用指定的 backbone 和提取的层
        feature_aggregator = wat.common.NetworkFeatureAggregator(
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

        #TODO: 选择预处理层
        # 使用 patch 级 Gram 预处理
        # preprocessing = wat.common.PatchGramPreprocessing(output_dim=pre_dim)
        # 配置预处理层：将每层展平后自适应池化到 pre_dim，再堆叠供后续聚合
        preprocessing = wat.common.Preprocessing(feature_dimensions, pre_dim)
        self.forward_modules["preprocessing"] = preprocessing

        # 配置嵌入层：目标嵌入维度用于训练后的特征空间转换
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = wat.common.Aggregator(target_dim=target_embed_dimension)

        # 将嵌入层（`preadapt_aggregator`）移动到指定的计算设备
        _ = preadapt_aggregator.to(self.device)

        # 将 `preadapt_aggregator` 添加到 `forward_modules` 中
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # 配置最近邻异常评分器：`NearestNeighbourScorer`，使用指定的最近邻方法和数量
        self.anomaly_scorer = wat.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn,
            nn_method=nn_method,
        )

        # 将采样器赋值给类的成员变量，默认使用 `IdentitySampler`（不进行特征采样）
        self.featuresampler = featuresampler

    def embed(self, data):
        """
        对输入数据计算 WAT 的图像级特征嵌入。
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
        将输入图像批次映射为图像级嵌入向量（不再做 patch 切分）。
        返回:
            features: np.ndarray 或 Tensor，形状 [B, target_embed_dimension]
            若 provide_patch_shapes=True 仍返回 (features, None) 以兼容旧调用。
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

        # 直接做层间预处理与聚合，得到每张图 1 个向量
        features = self.forward_modules["preprocessing"](features)   # [B, L, pre_dim]
        features = self.forward_modules["preadapt_aggregator"](features)  # [B, target_dim]

        if provide_patch_shapes:
            return _detach(features), None
        return _detach(features)

    def fit(self, training_data):
        """
        使用给定的训练数据构建 记忆库。
        
        参数:
            training_data :
                通常为 torch.utils.data.DataLoader，迭代输出训练图像。
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data, dataset_label_resolver=None):
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

        def _image_to_features(input_image, return_shapes=False):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image, provide_patch_shapes=return_shapes)

        features = []
        labels = []
        with tqdm.tqdm(
            input_data, desc="Computing features of memorybank ...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                meta = {}
                if isinstance(image, dict):
                    meta = image
                    image = image["image"]
                feats, _ = _image_to_features(image, return_shapes=True)

                # 解析标签：优先使用数据集返回的 generator（ai 用生成器类别，nature 统一为 nature）
                batch_dataset_names = []
                if isinstance(meta, dict) and meta.get("generator") is not None:
                    gen = meta.get("generator")
                    if isinstance(gen, (list, tuple)):
                        batch_dataset_names = [str(x) for x in gen]
                    else:
                        batch_dataset_names = [str(gen)] * image.shape[0]
                else:
                    # 兼容旧逻辑：从路径推断（默认取 ai 后一层目录）
                    batch_paths = meta.get("image_path") if isinstance(meta, dict) else None
                    if callable(dataset_label_resolver):
                        batch_dataset_names = dataset_label_resolver(meta, image, feats)
                    elif batch_paths is not None:
                        for p in batch_paths:
                            parts = str(p).replace("\\", "/").split("/")
                            lab = "unknown"
                            for i, part in enumerate(parts):
                                if part == "ai" and i + 1 < len(parts):
                                    # 优先支持 split/ai/<generator>/xxx.png
                                    cand = parts[i + 1]
                                    ext = os.path.splitext(cand)[1]
                                    if ext == "" and cand not in ("", ".", ".."):
                                        lab = cand
                                    # 兼容 split 的父目录为 generator：.../<generator>/<split>/ai/xxx.png
                                    elif i >= 2:
                                        lab = parts[i - 2]
                                    break
                            if lab == "unknown" and "nature" in parts:
                                lab = "nature"
                            batch_dataset_names.append(lab)
                    else:
                        batch_dataset_names = ["unknown"] * image.shape[0]

                # 展平后特征顺序与 _embed 一致：按图像顺序堆叠
                for name in batch_dataset_names:
                    labels.append(name)
                features.append(feats)

        # 拼接所有 batch 的特征，features.shape == [N_total, D]
        features = np.concatenate(features, axis=0)
        labels = np.asarray(labels).reshape(-1)

        features = self.featuresampler.run(features)
        # 若采样器提供了索引，同步裁剪标签
        sampler_indices = getattr(self.featuresampler, "last_indices", None)
        if sampler_indices is not None and sampler_indices is not slice(None):
            labels = labels[sampler_indices]

        # 对特征进行“建索引 / 建记忆库”，附带标签（ai bank 用 generator，nature bank 用 "nature"）
        self.anomaly_scorer.fit(detection_features=[features], detection_labels=labels)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def predict_with_meta(self, data):
        """
        与 predict 相同，但额外返回每张图像最近邻所属的标签与置信度。
        - DataLoader: 返回 (scores, masks, labels_gt, paths, nearest_labels, nearest_confs, gt_generators, gt_dataset_names)
        - Tensor:      返回 (scores, masks, nearest_labels, nearest_confs)
        """
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, return_meta=True)
        return self._predict(data, return_meta=True)

    def _predict_dataloader(self, dataloader, return_meta: bool = False):
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
        datasets_pred = []
        confs_pred = []
        generators_gt = []
        dataset_names_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                # 这里 batch 一定是 dict
                images = batch.get("image")
                batch_size = images.shape[0]
                batch_paths = batch.get("image_path")
                label_tensor = batch.get("is_ai")
                batch_generators = batch.get("generator")
                batch_dataset_names = batch.get("dataset_name")

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

                batch_scores, batch_masks, batch_datasets, batch_confs = (
                    self._predict(images, return_meta=True)
                    if return_meta
                    else self._predict(images, return_meta=False) + (None, None)
                )
                scores.extend(batch_scores)
                masks.extend(batch_masks)
                paths.extend(batch_paths)
                if return_meta:
                    # 真实的 generator/dataset_name 来自数据集（与最近邻预测无关）
                    if batch_generators is None:
                        generators_gt.extend(["unknown"] * batch_size)
                    elif isinstance(batch_generators, (list, tuple)):
                        generators_gt.extend([str(x) for x in batch_generators])
                    elif isinstance(batch_generators, torch.Tensor):
                        generators_gt.extend([str(x) for x in batch_generators.detach().cpu().numpy().tolist()])
                    else:
                        generators_gt.extend([str(batch_generators)] * batch_size)

                    if batch_dataset_names is None:
                        dataset_names_gt.extend(["unknown"] * batch_size)
                    elif isinstance(batch_dataset_names, (list, tuple)):
                        dataset_names_gt.extend([str(x) for x in batch_dataset_names])
                    elif isinstance(batch_dataset_names, torch.Tensor):
                        dataset_names_gt.extend([str(x) for x in batch_dataset_names.detach().cpu().numpy().tolist()])
                    else:
                        dataset_names_gt.extend([str(batch_dataset_names)] * batch_size)
                if return_meta and batch_datasets is not None:
                    datasets_pred.extend(batch_datasets)
                if return_meta and batch_confs is not None:
                    confs_pred.extend(batch_confs)

        if return_meta:
            return (scores, masks, labels_gt, paths, datasets_pred, confs_pred, generators_gt, dataset_names_gt)
        return (scores, masks, labels_gt, paths)

    def _predict(self, images, return_meta: bool = False):
        """
        对一批图像进行异常检测推理（无 patch，直接图像级 KNN）。

        返回:
            image_scores : list[float]  每张图的异常分数
            masks : list[None]         仅占位（已去除 Patch 级热力图）
        """

        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        # 获取 batch 大小（后面 reshape、聚合时要用）
        batchsize = images.shape[0]

        with torch.no_grad():
            # 提取图像的 patch 特征嵌入
            features, _ = self._embed(images, provide_patch_shapes=True)

            # 将特征转换为 numpy 数组，以便交给 FAISS/KNN 模块
            features = np.asarray(features)

            # 使用记忆库进行最近邻搜索
            predict_out = self.anomaly_scorer.predict([features])
            patch_scores = predict_out[0]
            nearest_datasets = predict_out[3] if len(predict_out) > 3 else None
            nearest_confs = predict_out[4] if len(predict_out) > 4 else None

            # 直接把每行结果视为图像分数
            image_scores = patch_scores.reshape(batchsize, -1).mean(axis=1)

            image_datasets = None
            if nearest_datasets is not None and len(nearest_datasets):
                image_datasets = [nearest_datasets[i] for i in range(len(image_scores))]
            image_confs = None
            if nearest_confs is not None and len(nearest_confs):
                image_confs = [float(nearest_confs[i]) for i in range(len(image_scores))]

        masks_placeholder = [None] * batchsize
        if return_meta:
            return [float(s) for s in image_scores], masks_placeholder, image_datasets, image_confs
        return [float(s) for s in image_scores], masks_placeholder


    @staticmethod
    def _params_file(filepath, prepend=""):
        """
        构造保存 WAT 配置参数的文件路径。
        参数:
            filepath : str
                保存目录（通常是某个实验 / 模型输出目录）。
            prepend : str, 可选
                文件名前缀，用于区分不同配置版本。
        返回:
            str: 完整的参数文件路径，例如：
                 "<filepath>/<prepend>wat_params.pkl"
        """
        return os.path.join(filepath, prepend + "wat_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        """
        将当前 WAT 模型的“记忆库 + 配置参数”保存到指定目录。

        内容包括两部分：
            1. anomaly_scorer 的索引数据（记忆库特征、FAISS 索引等）；
            2. WAT 本身的关键超参数（backbone、层名、patch 大小等），
               方便之后在同样配置下恢复模型。
        参数:
            save_path : str
                目标保存目录。
            prepend : str, 可选
                文件名前缀，用于在同一目录下区分多组 WAT 配置。
        """
        LOGGER.info("Saving WAT data.")

        # 1) 保存异常评分器（KNN / FAISS）的内部索引 / 特征
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )

        # 2) 组织 WAT 的关键配置参数，便于后续 load 时重建结构
        wat_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        # 写入一个 pickle 文件，保存上述配置字典
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(wat_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: wat.common.BruteNN(),
        prepend: str = "",
    ) -> None:
        """
        从指定目录恢复并初始化一个 WAT 实例。

        恢复内容包括：
            1. WAT 的结构与超参数配置（backbone 名称、提取层、patch 大小等）；
            2. anomaly_scorer 的近邻索引 / 记忆库特征（与保存时配套）。

        参数:
            load_path : str
                保存模型的目录（与 save_to_path 使用的目录一致）。
            device : torch.device
                当前运行设备（如 torch.device("cuda") 或 "cpu"）。
            nn_method : wat.common.FaissNN 或其子类
                用于最近邻搜索的实现，默认使用 FaissNN(on_gpu=False, num_workers=4)。
            prepend : str, 可选
                文件名前缀，用于从特定前缀版本中加载（与保存时的 prepend 对应）。
        """
        LOGGER.info("Loading and initializing WAT.")

        # 1) 读取保存的 WAT 参数字典（结构配置）
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            wat_params = pickle.load(load_file)

        # 2) 根据保存的 backbone 名称重新构建 backbone 模型
        wat_params["backbone"] = wat.backbones.load(
            wat_params["backbone.name"]
        )
        wat_params["backbone"].name = wat_params["backbone.name"]
        del wat_params["backbone.name"]

        # 3) 调用 WAT.load(...)，用读取到的参数在当前实例上完成初始化
        self.load(**wat_params, device=device, nn_method=nn_method)

        # 4) 恢复 anomaly_scorer 的索引 / 特征记忆库
        self.anomaly_scorer.load(load_path, prepend)

    def add_dataset(self, dataloader: torch.utils.data.DataLoader, dataset_name: str):
        """
        将一个新的数据集追加到现有的 KNN 记忆库。
        - 不重建已有索引，直接追加特征与标签。
        """
        if dataloader is None:
            return
        _ = self.forward_modules.eval()

        features = []
        labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f"Adding dataset {dataset_name}", leave=False):
                imgs = batch["image"] if isinstance(batch, dict) else batch
                feats, _ = self._embed(imgs.to(self.device), provide_patch_shapes=True)
                features.append(feats)
                labels.extend([dataset_name] * feats.shape[0])

        if features:
            features = np.concatenate(features, axis=0)
            labels = np.asarray(labels).reshape(-1)
            features = self.featuresampler.run(features)
            sampler_indices = getattr(self.featuresampler, "last_indices", None)
            if sampler_indices is not None and sampler_indices is not slice(None):
                labels = labels[sampler_indices]
            self.anomaly_scorer.add_features(features, labels)
