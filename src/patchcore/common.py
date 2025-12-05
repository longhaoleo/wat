import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import torch.nn as nn

METRIC = "l2"


# 加入L2归一化
def _l2_normalize_np(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)   # 按行求范数
    n = np.maximum(n, eps)
    return x / n


#### 近邻匹配 (Faiss近邻搜索)
class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4, metric: str = METRIC) -> None:
        """
        基于 FAISS 的最近邻搜索封装。
        功能：
            - 根据给定特征维度创建合适的 FAISS 索引（L2 或内积）；
            - 负责在 CPU / GPU 之间移动索引；
            - 对外提供统一的 fit / search 接口，供上层最近邻评分类使用。
        参数:
            on_gpu : bool, 默认 False
                - True  时：在 GPU 上构建/搜索 FAISS 索引（需要 GPU 支持）；
                - False 时：仅使用 CPU 版本。
            num_workers : int, 默认 4
                设置 FAISS 内部使用的线程数，用于加速相似度搜索。
            metric : str, 默认 METRIC（通常为 "l2"）
                距离度量方式：
                    - "l2"：欧氏距离；
                    - "ip"：Inner Product，内积（可用于余弦相似度等）。
        """
        # 设置 FAISS 使用的线程数（CPU 多线程加速）
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None   # 实际的 FAISS 索引对象
        self.metric = metric

    def _gpu_cloner_options(self):
        """构造用于 CPU 索引克隆到 GPU 时的配置对象。"""
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),  # GPU 资源句柄
                0,                              # 使用第 0 号 GPU
                index,
                self._gpu_cloner_options(),
            )
        return index

    def _index_to_cpu(self, index):
        if not self.on_gpu:
            # 当前模式已是 CPU，直接调用 gpu_to_cpu 是安全的（否则在 GPU 模式下可能无效）
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        """
        根据特征维度和度量方式，创建一个具体的 FAISS 索引实例，供后面的 fit / search 使用。
        参数:
            dimension : int
                特征向量维度 D。
        返回:
            一个可用于 add / search 的 FAISS 索引实例：
                - 若 metric == "ip"：使用内积索引 IndexFlatIP / GpuIndexFlatIP；
                - 否则（默认 "l2"）：使用欧氏距离索引 IndexFlatL2 / GpuIndexFlatL2。
        """
        # 内积距离（可用于余弦相似度等）
        if self.metric == "ip":
            if self.on_gpu:
                # GPU 版本的内积索引
                return faiss.GpuIndexFlatIP(
                    faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
                )
            # CPU 版本的内积索引
            return faiss.IndexFlatIP(dimension)

        # L2 距离（欧氏距离）
        if self.on_gpu:
            # GPU 版本的 L2 索引
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        # CPU 版本的 L2 索引
        return faiss.IndexFlatL2(dimension)


    def fit(self, features: np.ndarray) -> None:
        """
        将特征添加到FAISS搜索索引中。
        参数:
            features: 形状为 NxD 的数组 (N为样本数，D为特征维度)
        """
        # 如果之前已经有一个索引，先清空它，避免旧数据残留
        if self.search_index:
            self.reset_index()

        # 根据特征维度 D 创建一个合适的 FAISS 索引结构
        self.search_index = self._create_index(features.shape[-1])

        #（如有需要）对索引做训练，例如 IVFPQ 这类近似索引需要先 train
        self._train(self.search_index, features)

        # 把所有特征加入到索引里，作为“数据库向量”
        self.search_index.add(features)

    def _train(self, _index, _features):
        """训练索引（此处不需要额外训练）"""
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回最近邻搜索的距离和索引。
        参数:
            n_nearest_neighbours: int
                每个查询要返回多少个最近邻（K）。
            query_features: np.ndarray
                查询特征，形状 [N_query, D]。
            index_features: np.ndarray, 可选
                如果提供，则在“这批特征”上临时建索引并搜索；
                如果不提供，则使用事先通过 fit() 建好的 self.search_index。
        """
        # 情况 1：使用之前 fit 好的全局索引 self.search_index
        if index_features is None:
            # 直接在已有索引上做搜索
            # 返回 (distances, indices)，distances/indices 形状为 [N_query, K]
            return self.search_index.search(query_features, n_nearest_neighbours)

        # 情况 2：用户传入一批新的 index_features，临时建一个索引，再在上面搜索
        # 为这次搜索创建一个新的索引（维度 = index_features.shape[-1]）
        search_index = self._create_index(index_features.shape[-1])
        # 如有需要（IVFPQ 等近似索引），先训练索引结构
        self._train(search_index, index_features)
        # 把 index_features 加入这个临时索引
        search_index.add(index_features)
        # 在这个临时索引上对 query_features 做 KNN 搜索
        return search_index.search(query_features, n_nearest_neighbours)


    def save(self, filename: str) -> None:
        """将FAISS索引保存到文件"""
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        """从文件加载FAISS索引"""
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        """重置FAISS索引"""
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        """训练近似FAISS索引"""
        index.train(features)

    def _gpu_cloner_options(self):
        """使用float16减少GPU内存消耗"""
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        """创建近似FAISS索引 (IVFPQ) 以加速搜索"""
        quantizer = faiss.IndexFlatIP(dimension) if self.metric == "ip" else faiss.IndexFlatL2(dimension)
        metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        index = faiss.IndexIVFPQ(
            quantizer,
            dimension,
            512,  # 中心点数量
            64,   # 子量化器数量
            8,    # 每个代码的比特数
            metric_type,
        ) 
        return self._index_to_gpu(index)


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


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        """用于把 patch 级别的分数图，缩放/平滑到最终图像尺寸的分割器。"""
        self.device = device
        # 输出图像的目标边长（假设是方形图像，224x224）
        self.target_size = target_size
        # 高斯平滑的标准差，用于让热力图更平滑
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        """
        将 patch 级分数图转换为最终的像素级分割图（热力图）。

        输入:
            patch_scores:
                - 形状通常为 [B, H_p, W_p]，B 为 batch 大小，
                  H_p/W_p 为 patch 网格的高宽（例如 28x28）。
                - 可以是 numpy.ndarray 或 torch.Tensor。
        流程:
            1. 若输入是 numpy，则先转成 torch.Tensor；
            2. 搬到指定 device，并在通道维上 unsqueeze 成 [B, 1, H_p, W_p]；
            3. 使用双线性插值，将 (H_p, W_p) 上采样到目标大小 (target_size, target_size)；
            4. 去掉通道维，转回 numpy 数组；
            5. 对每张图做一次高斯滤波，使热力图更平滑。
        """
        with torch.no_grad():
            # 1) numpy -> torch（如有必要）
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            # 2) 搬到 device，添加通道维 -> [B, 1, H_p, W_p]
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            # 3) 双线性插值到目标尺寸 [B, 1, target_size, target_size]
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            # 4) 去掉通道维，搬回 CPU，转回 numpy -> [B, target_size, target_size]
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        # 5) 对每张分数图做高斯平滑，得到最终的 segmentation mask / 热力图
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """高效的网络特征提取"""

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
        """前向传播，提取特征"""
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
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        最近邻异常评分类，用于计算图像或像素的异常分数

        参数:
            n_nearest_neighbours: 最近邻的数量，用于判断异常像素
            nn_method: 使用的最近邻搜索方法（如FaissNN）
        """
        self.feature_merger = ConcatMerger()  # 用于合并特征的合并器

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        # 图像级别的最近邻搜索函数
        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        # 像素级别的最近邻搜索函数
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """
        调用最近邻方法的fit函数进行训练

        参数:
            detection_features: 训练图像的特征列表，每个特征对应于某个图像的特征向量
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,  # 合并所有训练图像的特征，分层 N ，每层将特征压成一维
        )
        self.detection_features = _l2_normalize_np(self.detection_features)
        self.nn_method.fit(self.detection_features)  # 用合并后的特征训练最近邻模型，一维向量


    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测异常分数
        参数:
            detection_query_features: 测试图像的特征列表
        """
        query_features = self.feature_merger.merge(
            query_features,  # 合并测试图像的特征
        )  # query_features：[N_patches_total, 512]
        query_features = _l2_normalize_np(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)  # 查询最近邻

        # 将距离转换为余弦相似度，再映射到 [0,1] 的异常得分
        metric = METRIC
        if metric == "ip":
            sims = query_distances  # 内积即相似度
        else:
            sims = 1.0 - (query_distances / 2.0)  # L2(单位向量) -> cos
        sims = np.clip(sims, -1.0, 1.0)
        anomaly_scores = (1.0 - sims.mean(axis=-1)) / 2.0  # 相似度高 -> 异常低
        return anomaly_scores, query_distances, query_nns

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
