"""
WAT 公共组件模块。

本文件对应 `PROJECT_DETAILED_COMMENTS.md` 第 4 节：
1) 检索器：`BruteNN` / `FaissNN`；
2) 特征处理：`Preprocessing` / `Aggregator` / `NetworkFeatureAggregator`；
3) 评分器：`NearestNeighbourScorer`（fit/predict/vote）。
"""

import copy
import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from wat.sampler import BaseSampler

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

METRIC = "l2"


# 加入L2归一化
def _l2_normalize_np(x, eps=1e-12):
    """按行做 L2 归一化，避免向量尺度差异主导 KNN 距离。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)   # 按行求范数
    n = np.maximum(n, eps)
    return x / n


class PCASampler(BaseSampler):
    """
    基于 PCA 的特征采样器：
      - 对特征做 PCA/SVD 分解，在主成分空间中计算每个样本的“能量”（前 k 个主成分系数的范数）；
      - 按能量从大到小选择前 p% 的样本。

    参数:
        percentage : float
            采样比例 (0,1)，即保留样本的比例；
        n_components : int 或 None
            在 PCA 空间中使用的主成分个数；
            为 None 时自动取 min(8, 特征维度)。
    """

    def __init__(self, percentage: float, n_components: int | None = None):
        super().__init__(percentage)
        self.n_components = n_components

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.percentage == 1:
            self.last_indices = slice(None)
            return features

        self._store_type(features)

        # 转为 numpy 做 PCA（仅用于计算索引，真正的子集从原始 features 中取）
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = np.asarray(features)

        N = X.shape[0]
        print(f'features number: {N}')
        if N <= 1:
            return features

        X = X.reshape(N, -1).astype(np.float32, copy=False)

        # 中心化
        X_mean = X.mean(axis=0, keepdims=True)
        X_centered = X - X_mean

        try:
            # SVD: X_centered = U diag(S) V^T
            U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        except Exception:
            # 数值失败时退化为随机采样
            print("数值失败时退化为随机采样")
            n_keep = max(1, int(N * self.percentage))
            n_keep = min(n_keep, N)
            idx_fallback = np.random.choice(N, size=n_keep, replace=False)
            self.last_indices = idx_fallback
            if isinstance(features, torch.Tensor):
                idx_t = torch.as_tensor(idx_fallback, dtype=torch.long, device=features.device)
                subset = features.index_select(0, idx_t)
            else:
                subset = features[idx_fallback]
            return self._restore_type(subset)

        # 选取前 k 个主成分
        k = self.n_components
        max_k = S.shape[0]
        if k is None or k <= 0 or k > max_k:
            k = min(8, max_k)

        # 每个样本在 PCA 空间中的坐标: Z = U * S
        Z = U[:, :k] * S[:k]
        scores = np.linalg.norm(Z, axis=1)

        # 选择得分最高的前 p% 样本
        n_keep = max(1, int(N * self.percentage))
        n_keep = min(n_keep, N)
        idx = np.argsort(scores)[-n_keep:]
        self.last_indices = idx

        # 根据索引从原始特征中取子集（保持类型/设备）
        if isinstance(features, torch.Tensor):
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=features.device)
            subset = features.index_select(0, idx_t)
        else:
            subset = features[idx]

        return self._restore_type(subset)


#### 近邻匹配 (可选 Faiss / 默认 Brute)
class BruteNN(object):
    """
    纯 numpy 版 KNN：无 faiss 依赖，小规模数据够用。
    距离支持 L2 / 内积，默认 L2。
    """
    def __init__(self, metric: str = METRIC) -> None:
        self.metric = metric
        self.index_features: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> None:
        self.index_features = np.asarray(features, dtype=np.float32)

    def reset_index(self):
        self.index_features = None

    def add(self, features: np.ndarray) -> None:
        if features is None or len(features) == 0:
            return
        feats = np.asarray(features, dtype=np.float32)
        if self.index_features is None:
            self.index_features = feats
        else:
            self.index_features = np.concatenate([self.index_features, feats], axis=0)

    def _pairwise(self, query, index):
        if self.metric == "ip":
            sims = query @ index.T
            # 更高相似度更近，将其转为“距离”形式：-sim
            dist = -sims
        else:  # l2
            q2 = (query ** 2).sum(axis=1, keepdims=True)
            i2 = (index ** 2).sum(axis=1, keepdims=True).T
            dist = q2 + i2 - 2.0 * query @ index.T
        return dist

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        # run(...) 统一返回 (nn_dists, nn_indices)，与 faiss.search 口径一致。
        index = np.asarray(index_features if index_features is not None else self.index_features, dtype=np.float32)
        query = np.asarray(query_features, dtype=np.float32)
        if index is None:
            raise RuntimeError("Index features are empty, call fit() first.")

        dists = self._pairwise(query, index)
        n = min(n_nearest_neighbours, index.shape[0])
        nn_indices = np.argpartition(dists, kth=n-1, axis=1)[:, :n]
        # 重新按距离排序
        gathered = np.take_along_axis(dists, nn_indices, axis=1)
        order = np.argsort(gathered, axis=1)
        nn_indices = np.take_along_axis(nn_indices, order, axis=1)
        nn_dists = np.take_along_axis(dists, nn_indices, axis=1)
        return nn_dists, nn_indices

    def save(self, filename: str) -> None:
        if self.index_features is None:
            return
        with open(filename, "wb") as f:
            pickle.dump(self.index_features, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.index_features = pickle.load(f)


class FaissNN(object):
    """
    可选的 FAISS 加速 KNN。若环境未安装 faiss，初始化时会报错。
    """
    def __init__(self, on_gpu: bool = False, num_workers: int = 4, metric: str = METRIC) -> None:
        if not HAS_FAISS:
            raise RuntimeError("faiss not installed; use BruteNN instead.")
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        self.metric = metric

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.metric == "ip":
            if self.on_gpu:
                return faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
            return faiss.IndexFlatIP(dimension)
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        # 每次 fit 都重建索引，确保不会混入旧实验特征。
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self.search_index.add(features.astype(np.float32))

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ):
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)
        search_index = self._create_index(index_features.shape[-1])
        search_index.add(index_features.astype(np.float32))
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        if self.search_index is None:
            return
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

    def add(self, features: np.ndarray) -> None:
        if features is None or len(features) == 0:
            return
        if self.search_index is None:
            self.fit(features)
            return
        self.search_index.add(features.astype(np.float32))


class _BaseMerger:
    def __init__(self):
        """特征合并基类（用于把多组特征按列拼接起来）"""

    def merge(self, features: list):
        """
        合并给定特征列表。
        使用场景（例如在 NearestNeighbourScorer 中）：
            - detection_features / query_features 可能是一个 list，
              里面每个元素是一组特征矩阵，如：
                  feature_k: 形状 [N, D_k]
            - 不同元素代表不同来源 / 不同层的特征。
        处理逻辑：
            1. 先对列表中的每个 feature 调用子类实现的 `_reduce`，
               做各自的展平 / 聚合（如平均池化或直接 flatten）；
            2. 然后在特征维度上（axis=1）把这些特征横向拼接，
               得到一个大的特征矩阵。
        参数:
            features : list[np.ndarray]
                特征列表，每个元素形状大致为 [N, D_k]，
                其中 N 为样本数，D_k 为该组特征的维度。
        返回:
            np.ndarray:
                合并后的特征矩阵，形状 [N, sum(D_k)]，
                即把多组特征在“列方向”拼在一起。
        """
        # 对列表中的每一个特征矩阵先做一次 _reduce（子类具体实现）
        features = [self._reduce(feature) for feature in features]
        # 再在列方向上拼接成一个大特征矩阵
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        """
        将特征从 [N, C, W, H] 压缩为 [N, C]，对空间维做平均池化。
        输入:
            features: np.ndarray
                形状 [N, C, W, H]，其中
                    N: 样本数
                    C: 通道数
                    W,H: 空间维度
        处理:
            - 先把空间维展平：reshape -> [N, C, W*H]
            - 然后在最后一维上取均值：mean(axis=-1)
              得到每个通道在整张特征图上的平均响应，把 W*H 个数压成 1 个数
        返回:
            np.ndarray:
                形状 [N, C]，每个样本每个通道一个标量（平均值）。
        """
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(axis=-1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        """
        将特征从 [N, C, W, H] 完全展平为 [N, C*W*H]。
        输入:
            features: np.ndarray
                形状 [N, C, W, H]
        处理:
            - 直接把后面所有维度压成一维：reshape(len(features), -1)
              等价于 reshape(N, C*W*H)。
        返回:
            np.ndarray:
                形状 [N, C*W*H]，每个样本是一个长向量，
                包含原特征图中所有通道和空间位置的信息。
        """
        return features.reshape(len(features), -1)


class PatchGramPreprocessing(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(PatchGramPreprocessing, self).__init__()
        """
        Patch 级 Gram 特征预处理模块。

        功能：
            - 输入：来自多个层的 patch 级特征，每个元素形状 [N, C_i, p, p]；
              其中 N = B * Gh_ref * Gw_ref，是“patch 个数”，每一行对应一个 patch。
            - 对每个 patch 独立计算通道间的 Gram 矩阵 G_i ∈ R^{C_i×C_i}；
            - 将 Gram 展平为向量，再用 1D 自适应平均池化映射到固定长度 output_dim；
            - 最终输出形状 [N, L, output_dim]，L 为使用的层数，与 self.layers_to_extract_from 对齐。

        参数:
            output_dim : int
                每个 patch 的 Gram 特征最终压缩到的维度，
                一般可以设为 pretrain_embed_dimension 或 target_embed_dimension。
        """
        self.output_dim = output_dim

    def forward(self, features):
        """
        前向传播：对每层的 patch 特征计算 Gram + 池化。
        输入:
            features : list[Tensor]
                长度 = L（层数），第 i 个元素形状为 [N, C_i, p, p]：
                    N   = B * Gh_ref * Gw_ref（所有图的 patch 总数）
                    C_i = 第 i 层通道数
                    p   = patchsize（例如 3）
        返回:
            Tensor:
                形状 [N, L, output_dim]：
                    N   = patch 数
                    L   = 层数
                    output_dim = 每个 patch 的 Gram 向量维度
        """
        gram_embeds = []

        for feat in features:
            # feat: [N, C, p, p]
            N, C, p_h, p_w = feat.shape
            P = p_h * p_w                  # patch 内空间位置数 (= p*p)

            # 先在空间维上展平:
            #   [N, C, p, p] -> [N, C, P]
            feat_flat = feat.reshape(N, C, P)

            # ========= 通道压缩：C -> C_reduced =========
            # 这里用自适应平均池化在通道维上做无参数压缩:
            #   feat_flat:        [N, C, P]
            #   转置后:           [N, P, C]   （最后一维是通道）
            #   adaptive_avg_pool1d(..., C_reduced) 在通道维上平均到 32 维
            #   再转回:           [N, C_reduced, P]
            C_reduced = min(C, 64)
            feat_flat_T = feat_flat.transpose(1, 2)          # [N, P, C]
            feat_flat_T = F.adaptive_avg_pool1d(
                feat_flat_T, C_reduced
            )                                                # [N, P, C_reduced]
            feat_reduced = feat_flat_T.transpose(1, 2)       # [N, C_reduced, P]

            # ========= 在压缩后的通道上算 Gram =========
            #   G = F_reduced F_reduced^T / P
            #   feat_reduced:         [N, C_reduced, P]
            #   feat_reduced^T:       [N, P, C_reduced]
            #   gram:                 [N, C_reduced, C_reduced]
            gram = torch.bmm(
                feat_reduced, feat_reduced.transpose(1, 2)
            )
            gram = gram / float(P)                           # 归一化

            # 展平 Gram 矩阵:
            #   [N, C_reduced, C_reduced] -> [N, 1, C_reduced*C_reduced]
            gram_flat = gram.reshape(
                N, 1, C_reduced * C_reduced
            )                                                # [N, 1, 1024] 若 C_reduced=32

            # 映射到目标维度 output_dim（例如设成 1024 时为恒等重采样）:
            #   [N, 1, C_reduced^2] -> [N, 1, output_dim] -> [N, output_dim]
            gram_embed = F.adaptive_avg_pool1d(
                gram_flat, self.output_dim
            ).squeeze(1)                                     # [N, output_dim]

            gram_embeds.append(gram_embed)                   # 当前层: [N, output_dim]

        # 按层堆叠: 列表长度 = L，每个元素 [N, output_dim]
        # stack 后: [L, N, output_dim]，再在 dim=1 上堆: [N, L, output_dim]
        return torch.stack(gram_embeds, dim=1)               # [N, L, output_dim]


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        """
        多层特征预处理模块。
        作用：
            - 对来自不同层（通道数可能不同）的特征，分别做一套相同的“维度规整”；
            - 把每一层的特征都映射到统一的维度 output_dim，方便后续跨层融合。
        参数:
            input_dims : list[int]
                每个元素对应一层的“原始特征维度”。
                （这里只是记录一下，用来对齐外部配置，不直接参与构建）
            output_dim : int
                预处理后的目标维度，即每层会被压缩/映射到的长度，
                在 PatchCore 里对应 pretrain_embed_dimension。
        """
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 为每一个输入层准备一个独立的 MeanMapper，
        # 预期输入形状为 [N, C_i, p, p]即[B * Gh_ref * Gw_ref, C, p, p]，输出为 [N, output_dim]
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)


    def forward(self, features):
        """
        前向传播，将特征通过预处理模块进行处理
        """
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        """
        将任意形状的特征向量，压缩/映射到固定长度 preprocessing_dim。
        作用：
            - 输入可以是 [N, C, p, p] 或 [N, D] 等，
              会先展平成长度为 D' 的一维向量；
            - 然后用 1D 自适应平均池化，把 D' 映射到固定长度 preprocessing_dim，
              起到“降维/规整维度”的作用。
        """
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """
        进行自适应平均池化处理，将特征映射为指定维度。
        输入:
            features: Tensor
                形状类似 [N, ...]，第 0 维是patch数（B*Gh_ref*Gw_ref），其余维度会全部展平。
        流程:
            1. reshape 为 [N, 1, D']，D' = 其余所有维度展平后的长度；
            2. 对长度为 D' 的序列做自适应平均池化，输出长度为 preprocessing_dim；
            3. 去掉中间的通道维，得到 [N, preprocessing_dim]。
        返回:
            Tensor: 形状 [N, preprocessing_dim]。
        """
        # [N, ...] -> [N, 1, D']
        features = features.reshape(len(features), 1, -1)
        # 自适应平均池化: [N, 1, D'] -> [N, 1, preprocessing_dim] -> [N, preprocessing_dim]
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        """
        特征聚合模块：将输入特征统一压缩到固定长度 target_dim。
        场景：
            - 输入通常是来自多个层、已经预处理好的特征，
              形状一般为 [N, L, D] 或 [N, D']；
            - 这里会把后面的维度全部展平，再用 1D 自适应平均池化
              聚合成长度为 target_dim 的向量，作为最终的嵌入表示。
        """
        self.target_dim = target_dim

    def forward(self, features):
        """
        对输入特征做自适应平均池化聚合，并映射到目标维度。
        输入:
            features: Tensor
                形状 [N, ...]：
                    - N 是样本数（ N = B * Gh * Gw）；
                    - 后面的维度可能包含“层数 × 预处理维度”等信息。
        流程:
            1. 展平除 batch 维以外的所有维度：
               features -> [N, 1, D']；
            2. 对长度为 D' 的序列做自适应平均池化，
               输出固定长度 target_dim： [N, 1, target_dim]；
            3. 去掉中间的通道维，得到 [N, target_dim]。
        返回:
            Tensor: 形状 [N, target_dim]，
                    即每个样本一个最终聚合后的嵌入向量。
        """
        # [N, ...] -> [N, 1, D']
        features = features.reshape(len(features), 1, -1)
        # 自适应平均池化到 target_dim: [N, 1, D'] -> [N, 1, target_dim]
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        # 展平成 [N, target_dim]
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """
    网络中间层特征提取器。

    通过 forward hook 抓取指定层输出，并在最后一层后提前中断前向，减少无效计算。
    """

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """提取网络特征

        参数:
            backbone: torchvision 模型
            layers_to_extract_from: 要提取特征的层列表
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        print([f'层次结构：{backbone.__dict__["_modules"].keys()}'])

        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}  # 用于存储中间特征

        # 按配置逐层注册 hook，输出存到 self.outputs。
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )

            # 解析层级名称：递归获取每一层
            network_layer = self.get_layer_by_name(backbone, extract_layer)

            # 注册 hook
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        
        self.to(self.device)

    def get_layer_by_name(self, model, layer_name):
        """
        根据给定的层名称，递归查找并返回层。
        支持多层嵌套结构，如 'features.3.3'。
        """
        layers = layer_name.split(".")  # 使用 '.' 分割层名
        layer = model

        for layer_part in layers:
            if isinstance(layer, torch.nn.Sequential):
                # 如果是 Sequential 类型，按索引访问
                layer = layer[int(layer_part)]
            else:
                # 否则按名字访问
                layer = layer.__dict__["_modules"].get(layer_part)
                if layer is None:
                    print(f'当前层结构：{layer.__dict__["_modules"].keys()}')
                    raise ValueError(f"层 '{layer_part}' 未在当前网络结构中找到")
        
        return layer



    def forward(self, images):
        """执行 backbone 前向并收集 hook 输出。"""
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """计算每层的特征维度"""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        """
        用于从指定层提取特征的钩子类初始化

        参数:
            hook_dict: 用于存储层输出的字典
            layer_name: 当前层的名称
            last_layer_to_extract: 最后一层提取的层名，用于控制提取结束时抛出异常
        """
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract  # 如果当前层是最后一层，标记抛出异常
        )

    def __call__(self, module, input, output):
        """
        当经过该层时，保存该层的输出

        参数:
            module: 当前模块
            input: 输入
            output: 输出
        """
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            # 当达到最后一层时，抛出自定义异常停止计算
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    """自定义异常，用于标记最后一层提取结束"""
    pass


class NearestNeighbourScorer(object):
    def __init__(
        self,
        n_nearest_neighbours: int,
        nn_method=None,
        *,
        label_count_power: float = 0.5,
        weight_eps: float = 1e-12,
        diversity_alpha_entropy: float = 0.6,
        diversity_alpha_unique: float = 0.4,
    ) -> None:
        """
        最近邻异常评分类，用于计算图像或像素的异常分数

        参数:
            n_nearest_neighbours: 最近邻的数量，用于判断异常像素
            nn_method: 使用的最近邻搜索方法（如 FaissNN 或 BruteNN）
        """
        self.feature_merger = ConcatMerger()  # 用于合并特征的合并器

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method or BruteNN(METRIC)
        self.label_count_power = float(label_count_power)
        self.weight_eps = float(weight_eps)
        # TopK 内标签越“杂”，置信度越低；这里把惩罚作用在 conf 上（不改变 argmax label）。
        self.diversity_alpha_entropy = float(diversity_alpha_entropy)
        self.diversity_alpha_unique = float(diversity_alpha_unique)

        # 与 detection_features 行对齐的数据集标签
        self.detection_dataset_labels: Optional[np.ndarray] = None

        # 图像级别的最近邻搜索函数。
        # 注意这里使用 self.n_nearest_neighbours（而不是构造入参的闭包常量），
        # 以支持运行时调整 top-k。
        self.imagelevel_nn = lambda query: self.nn_method.run(
            self.n_nearest_neighbours, query
        )
        # 像素级别的最近邻搜索函数
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(
        self,
        detection_features: List[np.ndarray],
        detection_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        调用最近邻方法的fit函数进行训练

        参数:
            detection_features: 训练图像的特征列表，每个特征对应于某个图像的特征向量
            detection_labels: 可选，每个特征行对应的数据集名称
        """
        # 1) 把多组特征合并为单个二维矩阵 [N, D_total]。
        self.detection_features = self.feature_merger.merge(
            detection_features,  # 合并所有训练图像的特征，分层 N ，每层将特征压成一维
        )
        # 2) L2 归一化后再建索引，确保距离可比。
        self.detection_features = _l2_normalize_np(self.detection_features)
        if detection_labels is None:
            detection_labels = np.array(["unknown"] * len(self.detection_features))
        detection_labels = np.asarray(detection_labels).reshape(-1)
        if detection_labels.shape[0] != self.detection_features.shape[0]:
            raise ValueError("detection_labels length must match detection_features rows.")
        self.detection_dataset_labels = detection_labels
        # 3) 训练近邻索引（Brute 或 Faiss）。
        self.nn_method.fit(self.detection_features)


    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
        List[float],
        List[int],
        List[int],
        List[float],
        List[float],
        List[float],
        List[float],
    ]:
        """
        预测异常分数
        参数:
            detection_query_features: 测试图像的特征列表
        """
        # 1) 查询特征使用与训练同样的 merge + normalize 口径。
        query_features = self.feature_merger.merge(
            query_features,  # 合并测试图像的特征
        )  # query_features：[N_patches_total, 512]
        query_features = _l2_normalize_np(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)  # 查询最近邻

        # 2) 异常分数定义：TopK L2 距离均值，再映射到 [0,1]。
        #    归一化向量的 L2 距离理论范围约 [0, 2]，所以这里除以 2。
        mean_dists = query_distances.mean(axis=-1)
        anomaly_scores = np.clip(mean_dists / 2.0, 0.0, 1.0)

        pred_labels: List[str] = []
        pred_label_confs: List[float] = []
        pred_label_counts: List[int] = []
        pred_label_unique: List[int] = []
        pred_label_base_confs: List[float] = []
        pred_label_diversity_penalties: List[float] = []
        pred_label_entropy_normalized: List[float] = []
        pred_label_unique_ratios: List[float] = []
        # 3) 若索引保存了标签，则额外返回 TopK 投票标签与置信度。
        if self.detection_dataset_labels is not None:
            for row_idx, nn_row in enumerate(query_nns):
                labels = self.detection_dataset_labels[nn_row]
                dists = query_distances[row_idx]
                (
                    best_label,
                    conf,
                    best_count,
                    n_unique,
                    base_conf,
                    diversity_penalty,
                    entropy_normalized,
                    unique_ratio,
                ) = self._vote_label(labels=labels, dists=dists)
                pred_labels.append(best_label)
                pred_label_confs.append(conf)
                pred_label_counts.append(best_count)
                pred_label_unique.append(n_unique)
                pred_label_base_confs.append(base_conf)
                pred_label_diversity_penalties.append(diversity_penalty)
                pred_label_entropy_normalized.append(entropy_normalized)
                pred_label_unique_ratios.append(unique_ratio)

        return (
            anomaly_scores,
            query_distances,
            query_nns,
            pred_labels,
            pred_label_confs,
            pred_label_counts,
            pred_label_unique,
            pred_label_base_confs,
            pred_label_diversity_penalties,
            pred_label_entropy_normalized,
            pred_label_unique_ratios,
        )

    def _vote_label(
        self,
        *,
        labels: np.ndarray,
        dists: np.ndarray,
    ) -> tuple[str, float, int, int, float, float, float, float]:
        """
        TopK label 投票（距离权重 + 数量加成）并返回惩罚后的置信度。
        返回:
          (
            best_label,
            conf_after_penalty,
            best_count,
            n_unique,
            base_conf_before_penalty,
            diversity_penalty,
            entropy_normalized,
            unique_ratio,
          )
        """
        # 先把距离转成权重：越近权重越大。
        weights = self._weights_from_dists(dists)

        weight_sum_by_label: Dict[str, float] = {}
        count_by_label: Dict[str, int] = {}
        for lab, w in zip(labels, weights):
            k = str(lab)
            weight_sum_by_label[k] = weight_sum_by_label.get(k, 0.0) + float(w)
            count_by_label[k] = count_by_label.get(k, 0) + 1

        if not weight_sum_by_label:
            return "unknown", float("nan"), 0, 0, float("nan"), float("nan"), float("nan"), float("nan")

        # 距离权重与出现次数共同决定分数：频次高且距离近的标签更优。
        scored = {
            k: weight_sum_by_label[k] * (count_by_label[k] ** self.label_count_power)
            for k in weight_sum_by_label.keys()
        }
        best_label = max(scored, key=scored.get)
        best_score = float(scored[best_label])
        denom = float(sum(scored.values())) + self.weight_eps
        base_conf = best_score / denom

        # 邻居标签越杂，置信度惩罚越强（但不会改变 argmax 标签）。
        penalty, entropy_normalized, unique_ratio = self._diversity_penalty(scored=scored, k=len(dists))
        conf = float(np.clip(base_conf * penalty, 0.0, 1.0))
        return (
            best_label,
            conf,
            int(count_by_label.get(best_label, 0)),
            int(len(count_by_label)),
            float(base_conf),
            float(penalty),
            float(entropy_normalized),
            float(unique_ratio),
        )

    def _weights_from_dists(self, dists: np.ndarray) -> np.ndarray:
        # 仅使用 L2 距离：距离越小，权重越大。
        return 1.0 / (dists.astype(np.float64) + self.weight_eps)

    def _diversity_penalty(self, *, scored: Dict[str, float], k: int) -> tuple[float, float, float]:
        """
        计算“杂乱度”惩罚因子（用于降低最近邻标签置信度）：
          - entropy：score 分布越均匀，越杂
          - unique_ratio：TopK 里不同 label 越多，越杂

        返回:
          (penalty, entropy_norm, unique_ratio)
        """
        n_unique = len(scored)
        if n_unique <= 1:
            return 1.0, 0.0, 0.0

        values = np.asarray(list(scored.values()), dtype=np.float64)
        total = float(values.sum())
        if not np.isfinite(total) or total <= self.weight_eps:
            # 极端数值场景：不做惩罚，避免把 conf 直接压没
            return 1.0, 0.0, 0.0

        p = values / (total + self.weight_eps)
        ent = -float(np.sum(p * np.log(p + self.weight_eps)))
        ent_max = float(np.log(n_unique + self.weight_eps))
        entropy_norm = 0.0 if ent_max <= 0.0 else float(np.clip(ent / ent_max, 0.0, 1.0))

        # 归一化 unique_ratio：n_unique=1 -> 0，n_unique=k -> 1
        unique_ratio = float(np.clip((n_unique - 1) / float(max(1, k - 1)), 0.0, 1.0))

        # 惩罚乘法组合：两者任何一个很“杂”都会降低 penalty
        p_ent = 1.0 - self.diversity_alpha_entropy * entropy_norm
        p_uni = 1.0 - self.diversity_alpha_unique * unique_ratio
        penalty = float(np.clip(p_ent, 0.0, 1.0) * np.clip(p_uni, 0.0, 1.0))
        return penalty, entropy_norm, unique_ratio

    @staticmethod
    def _detection_file(folder, prepend=""):
        """
        获取存储检测特征的文件路径

        参数:
            folder: 文件夹路径
            prepend: 文件名前缀
        """
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        """
        获取存储搜索索引的文件路径

        参数:
            folder: 文件夹路径
            prepend: 文件名前缀
        """
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        """
        保存特征到文件

        参数:
            filename: 保存文件的路径
            features: 特征数据
        """
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        """
        从文件加载特征

        参数:
            filename: 文件路径
        """
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        """
        保存模型和特征到指定文件夹

        参数:
            save_folder: 保存文件夹路径
            save_features_separately: 是否单独保存特征
            prepend: 文件名前缀
        """
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )
        if self.detection_dataset_labels is not None:
            self._save(
                os.path.join(save_folder, prepend + "nnscorer_labels.pkl"),
                self.detection_dataset_labels,
            )

    def save_and_reset(self, save_folder: str) -> None:
        """保存并重置最近邻索引"""
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        """从指定文件夹加载模型和特征"""
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
        labels_path = os.path.join(load_folder, prepend + "nnscorer_labels.pkl")
        if os.path.exists(labels_path):
            self.detection_dataset_labels = self._load(labels_path)
        else:
            self.detection_dataset_labels = None

    def add_features(
        self, new_features: np.ndarray, new_labels: Optional[np.ndarray] = None
    ) -> None:
        """
        向现有索引追加新特征，并保存对应数据集标签。
        """
        if new_features is None or len(new_features) == 0:
            return
        new_features = _l2_normalize_np(new_features)
        self.nn_method.add(new_features)
        if getattr(self, "detection_features", None) is None:
            self.detection_features = new_features
        else:
            self.detection_features = np.concatenate(
                [self.detection_features, new_features], axis=0
            )

        if new_labels is None:
            new_labels = np.array(["unknown"] * len(new_features))
        else:
            new_labels = np.asarray(new_labels).reshape(-1)
            if len(new_labels) != len(new_features):
                raise ValueError("new_labels length must match new_features rows.")

        if self.detection_dataset_labels is None:
            self.detection_dataset_labels = new_labels
        else:
            self.detection_dataset_labels = np.concatenate(
                [self.detection_dataset_labels, new_labels], axis=0
            )
