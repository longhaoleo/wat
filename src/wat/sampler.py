import abc
from typing import Union

import numpy as np
import torch
import tqdm

# 定义一个身份采样器，不进行任何处理，直接返回输入特征
class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        self.last_indices = slice(None)  # 记录未采样
        return features


# 基础采样器类，其他采样器可以继承它并实现特定的采样逻辑
class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        """
        初始化基础采样器
        参数:
            percentage: 采样的比例 (0, 1) 范围内的浮动值，表示采样的比例。
        """
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage
        self.last_indices = None

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        子类必须实现的抽象方法，负责采样特征
        参数:
            features: 输入特征，类型可以是 torch.Tensor 或 np.ndarray。
        返回:
            采样后的特征，类型与输入相同。
        """
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        """
        存储特征的数据类型（是否为NumPy数组，或者是PyTorch张量）
        参数:
            features: 输入特征
        """
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features) -> Union[torch.Tensor, np.ndarray]:
        """
        恢复特征的数据类型，确保输出与输入一致。
        - 若原始输入是 numpy，则返回 numpy（必要时从 Tensor 转回）。
        - 若原始输入是 Tensor，则返回 Tensor（必要时从 numpy 转为 Tensor 并放回原设备）。
        """
        if self.features_is_numpy:
            if isinstance(features, np.ndarray):
                return features
            if isinstance(features, torch.Tensor):
                return features.detach().cpu().numpy()
            return np.asarray(features)
        # 原始输入为 Tensor
        if isinstance(features, torch.Tensor):
            return features.to(self.features_device)
        return torch.as_tensor(features).to(self.features_device)


# 贪心核心集采样器类，使用贪心算法来选择样本子集
class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """
        贪心核心集采样器初始化

        参数:
            percentage: 采样比例，表示从输入数据中采样多少比例的数据
            device: 设备（CPU或GPU），用于存储和计算
            dimension_to_project_features_to: 目标维度，数据将投影到这个维度
        """
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        """
        将特征投影到目标维度

        参数:
            features: 输入特征
        
        返回:
            投影后的特征
        """
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        使用贪心核心集方法进行特征采样

        参数:
            features: 输入特征，形状为 [N x D]，N为样本数，D为特征维度
        
        返回:
            采样后的特征
        """
        if self.percentage == 1:
            self.last_indices = slice(None)
            return features
        self._store_type(features)  # 存储特征的数据类型
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)  # 降维处理
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)  # 获取贪心采样的索引
        self.last_indices = sample_indices
        features = features[sample_indices]  # 根据索引选择样本
        return self._restore_type(features)  # 恢复特征的数据类型

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个矩阵之间的欧几里得距离

        参数:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵
        
        返回:
            两个矩阵的欧几里得距离矩阵
        """
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """
        运行贪心核心集选择算法，选择最有代表性的样本

        参数:
            features: 输入特征，形状为 [NxD]
        
        返回:
            采样后的索引
        """
        distance_matrix = self._compute_batchwise_differences(features, features)  # 计算距离矩阵
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)  # 计算核心集的距离

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)  # 计算需要的样本数

        # 使用 tqdm 显示贪心核心集选择的进度
        for _ in tqdm.tqdm(range(num_coreset_samples), desc="Greedy coreset subsampling"):
            select_idx = torch.argmax(coreset_anchor_distances).item()  # 选择距离最大的样本
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # 获取选择样本的距离
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values  # 更新距离

        return np.array(coreset_indices)  # 返回最终选择的核心集样本索引


# 近似贪心核心集采样器类，改进了贪心核心集采样，减少内存消耗
class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """
        初始化近似贪心核心集采样器

        参数:
            percentage: 采样比例
            device: 设备（CPU或GPU）
            number_of_starting_points: 初始起始点数量
            dimension_to_project_features_to: 目标维度
        """
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """
        运行近似贪心核心集选择，减少内存消耗

        参数:
            features: 输入特征
        
        返回:
            近似贪心核心集样本的索引
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


# 随机采样器类，从特征集中随机选择样本
class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """从特征集中随机选择样本

        参数:
            features: [N x D] 输入特征
        
        返回:
            随机选择的特征
        """
        # 处理前的特征规模
        n_before = len(features)
        shape_before = getattr(features, "shape", None)

        # 随机采样索引
        num_random_samples = int(n_before * self.percentage)
        subset_indices = np.random.choice(n_before, num_random_samples, replace=False)
        subset_indices = np.array(subset_indices)
        self.last_indices = subset_indices

        # 选择子集
        subset = features[subset_indices]
        shape_after = getattr(subset, "shape", None)

        # 打印处理前后特征信息（数量 / 形状），便于快速确认采样效果
        try:
            print(
                f"[RandomSampler] features: N={n_before}, shape={shape_before} -> "
                f"N={len(subset)}, shape={shape_after}"
            )
        except Exception:
            # 打印不是关键路径，若环境不允许打印，忽略异常
            pass

        return subset


# -----------------------------
# Central/Core Sampler：尽量保留主体（中心）
# -----------------------------
class CentralSampler(BaseSampler):
    def __init__(self, percentage: float, use_mahalanobis: bool = False, epsilon: float = 1e-6):
        super().__init__(percentage)
        self.use_mahalanobis = use_mahalanobis
        self.epsilon = float(epsilon)

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        中心采样：
          - 计算每个样本到整体中心的距离（L2 或 Mahalanobis），
          - 按距离升序选取前 p%（越靠近中心越优先）。
        """
        with tqdm.tqdm(total=4, desc="CentralSampler", leave=False) as pbar:
            self._store_type(features)
            pbar.update()
            X = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
            X = X.reshape(X.shape[0], -1)
            N = X.shape[0]
            k = max(1, int(N * self.percentage))
            pbar.update()

            mu = X.mean(axis=0, keepdims=True)
            if self.use_mahalanobis:
                S = np.cov((X - mu).T) + self.epsilon * np.eye(X.shape[1])
                try:
                    S_inv = np.linalg.inv(S)
                except Exception:
                    S_inv = np.diag(1.0 / (np.diag(S) + self.epsilon))
                diff = X - mu
                dist = np.einsum("ni,ij,nj->n", diff, S_inv, diff)
            else:
                dist = np.linalg.norm(X - mu, axis=1)
            pbar.update()

            idx = np.argsort(dist)[:k]
            self.last_indices = idx
            subset = features[idx]
            pbar.update()

        try:
            print(f"[CentralSampler] selected {len(idx)} / {N} (p={self.percentage}, mahal={self.use_mahalanobis}).")
        except Exception:
            pass
        return self._restore_type(subset)


# -----------------------------
# Density Sampler：按密度（kNN 平均距离）保留主体
# -----------------------------
class DensitySampler(BaseSampler):
    def __init__(self, percentage: float, n_neighbors: int = 10):
        super().__init__(percentage)
        self.n_neighbors = max(2, int(n_neighbors))

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        密度采样：
          - 计算每个点到其 k 个最近邻的平均距离，平均距离越小密度越高；
          - 按平均距离升序选取前 p%。
        """
        with tqdm.tqdm(total=5, desc="DensitySampler", leave=False) as pbar:
            self._store_type(features)
            pbar.update()
            X = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
            X = X.reshape(X.shape[0], -1)
            N = X.shape[0]
            k_pick = max(1, int(N * self.percentage))
            k = min(self.n_neighbors, max(2, N))
            pbar.update()

            try:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
                nn.fit(X)
                pbar.update()
                dists, _ = nn.kneighbors(X, n_neighbors=k, return_distance=True)
                mean_dist = dists[:, 1:].mean(axis=1) if dists.shape[1] >= 2 else dists[:, 0]
            except Exception as e:
                tqdm.tqdm.write(f"[DensitySampler] sklearn NearestNeighbors failed ({e}); falling back to Random.")
                idx = np.random.choice(N, k_pick, replace=False)
                pbar.close()
                return self._restore_type(features[idx])

            pbar.update()
            idx_sorted = np.argsort(mean_dist)[:k_pick]
            self.last_indices = idx_sorted
            subset = features[idx_sorted]
            pbar.update()

        try:
            print(f"[DensitySampler] selected {len(idx_sorted)} / {N} (p={self.percentage}, kNN={k}).")
        except Exception:
            pass
        return self._restore_type(subset)
# KMeans / MiniBatchKMeans 采样器：以簇中心为代表点，保证代表性
class KMeansSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        max_iter: int = 100,
        batch_size: int = 1024,
        random_state: int = 0,
        per_cluster_topk: int = 1,
    ):
        """
        基于 KMeans 的代表性采样器：
          - 将特征聚为 k 个簇（k≈percentage*N），每簇选取离中心最近的样本作为代表。

        参数：
          percentage   采样比例（0,1）
          max_iter     KMeans 最大迭代次数
          batch_size   MiniBatchKMeans 的 batch 大小
          random_state 随机种子
        """
        super().__init__(percentage)
        self.use_minibatch = True  # 默认使用 MiniBatchKMeans
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.per_cluster_topk = max(1, int(per_cluster_topk))

    def _fit_kmeans(self, X: np.ndarray, n_clusters: int):
        try:
            if self.use_minibatch:
                from sklearn.cluster import MiniBatchKMeans as _K
                kmeans = _K(
                    n_clusters=n_clusters,
                    batch_size=min(self.batch_size, max(100, n_clusters * 10)),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init="auto",
                )
            else:
                from sklearn.cluster import KMeans as _K
                kmeans = _K(
                    n_clusters=n_clusters,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init="auto",
                )
            kmeans.fit(X)
            return kmeans
        except Exception as e:
            # 若 sklearn 不可用或失败，退化为随机采样
            tqdm.tqdm.write(f"[KMeansSampler] sklearn clustering failed ({e}); falling back to Random.")
            idx = np.random.choice(len(X), n_clusters, replace=False)
            class Fallback:
                cluster_centers_ = X[idx]
                labels_ = np.arange(len(X)) % n_clusters
            return Fallback()

    def sample_indices(self, features: Union[torch.Tensor, np.ndarray], n: int | None = None) -> np.ndarray:
        # 转为 numpy 并展平到 [N, D]
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = np.asarray(features)
        X = X.reshape(X.shape[0], -1)

        n_total = X.shape[0]
        n_clusters = int(n if (n is not None) else max(1, int(n_total * self.percentage)))
        n_clusters = max(1, min(n_clusters, n_total))
        if n_clusters >= n_total:
            return np.arange(n_total, dtype=np.int64)

        # 训练 KMeans
        kmeans = self._fit_kmeans(X, n_clusters=n_clusters)
        centers = np.asarray(kmeans.cluster_centers_)
        labels = np.asarray(kmeans.labels_)

        # 每簇选择离中心最近的前 topk 个样本（Cluster-TopC）
        selected = []
        # 使用 tqdm 显示按簇选择代表点的进度
        for c in tqdm.tqdm(range(n_clusters), desc="KMeans cluster-topk"):
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0:
                continue
            Xc = X[idx_c]
            d = np.linalg.norm(Xc - centers[c], axis=1)
            order = np.argsort(d)[: self.per_cluster_topk]
            selected.extend(idx_c[order].tolist())

        if len(selected) < n_clusters:
            # 如果某些簇为空，随机补齐到目标数
            remaining = np.setdiff1d(np.arange(n_total), np.asarray(selected, dtype=np.int64), assume_unique=False)
            need = n_clusters - len(selected)
            if need > 0 and remaining.size > 0:
                pick = np.random.choice(remaining, size=min(need, remaining.size), replace=False)
                selected.extend(pick.tolist())

        return np.asarray(selected, dtype=np.int64)

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """执行 KMeans 采样并返回子集特征，类型与设备与输入一致。"""
        self._store_type(features)
        idx = self.sample_indices(features)
        subset = features[idx]
        try:
            print(
                f"[KMeansSampler] selected {len(idx)} / {len(features)} samples "
                f"(p={self.percentage}, topk={self.per_cluster_topk}, minibatch={self.use_minibatch})."
            )
        except Exception:
            pass
        return self._restore_type(subset)
