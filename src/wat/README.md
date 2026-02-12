## WAT 代码结构（精简版）

本仓库当前实现的是 **图像级 embedding + KNN** 的 WAT 流程（不做 PatchMaker / 分割可视化 / 逻辑回归分类器）。

### 核心文件

1. `wat.py`
   - `WAT`：主模型类
   - `fit()`：构建 memorybank（提特征 -> 采样 -> 建 KNN 索引）
   - `predict()/predict_with_meta()`：推理，输出图像分数，并可附带最近邻的 generator 预测与置信度

2. `common.py`
   - `BruteNN`：纯 numpy KNN（默认）
   - `FaissNN`：可选 faiss 后端（安装了 faiss 才可用）
   - `NetworkFeatureAggregator/Preprocessing/Aggregator`：多层特征提取与规整
   - `NearestNeighbourScorer`：KNN 打分 + generator 投票（距离权重 + 数量加成 + 杂乱度惩罚）

3. `backbones.py`
   - `load()`：加载 backbone（包含 open_clip 的视觉 backbone）

4. `sampler.py`
   - `IdentitySampler/RandomSampler/...`：memorybank 特征采样（可选降采样以控制索引大小）

5. `datasets/`
   - `tiny_genimage.py`：Tiny-GenImage 风格数据集读取；支持从目录层级提供 `generator`/`dataset_name`
