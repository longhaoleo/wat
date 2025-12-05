
## 模块依赖关系

本项目由多个独立模块组成，主要模块间的依赖关系如下：

1. **`common.py`**：提供了各种公共功能，如近邻搜索（FAISS）、特征合并等基础工具。它被多个模块依赖，主要用于提供常用的工具函数和类。
2. **`patchcore.py`**：实现了 PatchCore 异常检测方法，是核心检测模块，依赖于 `common.py` 中的工具类，如 FAISS、特征聚合和最近邻计算等。
3. **`metrics.py`**：提供了用于评估异常检测结果的各种度量函数，包括 AUROC、FPR、TPR 等，用于评估模型在图像和像素级的检测表现。
4. **`sampler.py`**：实现了不同的采样方法，包括 `GreedyCoresetSampler` 和 `RandomSampler` 等，用于在训练过程中采样数据集，减少冗余并提高计算效率。它主要被 `patchcore.py` 中用于特征采样。
5. **`backbones.py`**：提供了多个常见的深度学习模型（如 ResNet, VGG 等）作为特征提取的骨干网络（backbone），这些网络被 `patchcore.py` 和其他模块用于提取图像特征。
6. **`utils.py`**：提供了各种实用的工具函数，如图像保存、设备设置、随机种子固定等，用于简化工作流并提高可复现性。

## 文件功能细节

### `common.py`

**功能**：提供常用的工具类和函数。

* **FaissNN**：实现了使用 FAISS 库进行最近邻搜索的功能，可以在 CPU 或 GPU 上执行。
* **ApproximateFaissNN**：继承自 `FaissNN`，提供了使用 IVFPQ 算法进行近似最近邻搜索的功能，适用于大规模数据。
* **_BaseMerger, AverageMerger, ConcatMerger**：用于特征合并，支持将多个特征进行合并操作（平均合并或拼接合并）。
* **Preprocessing, MeanMapper, Aggregator**：特征预处理模块，包括对特征的降维、聚合操作等。
* **RescaleSegmentor**：将异常分数转换为分割掩码，并进行高斯平滑处理。
* **NetworkFeatureAggregator**：通过卷积神经网络提取中间层特征，支持多个网络层的特征提取。
* **NearestNeighbourScorer**：通过最近邻方法计算图像和像素的异常分数，用于异常检测。

### `patchcore.py`

**功能**：实现 PatchCore 异常检测方法。

* **PatchCore**：核心异常检测类，使用 PatchCore 方法对图像进行异常检测。包括以下功能：

  * 加载预训练的骨干网络和其他模块。
  * 提取图像特征并进行补丁分割。
  * 使用最近邻方法进行异常评分。
  * 训练和预测功能。
  * 支持通过内存银行和核心集子采样来提高效率。
  * 支持图像级和像素级异常检测。
  * 模型的保存和加载。

### `metrics.py`

**功能**：提供用于异常检测任务的评估指标计算。

* **compute_imagewise_retrieval_metrics**：计算图像级别的评估指标，包括 AUROC、精度、召回率、F1 分数等。
* **compute_pixelwise_retrieval_metrics**：计算像素级别的评估指标，包括 AUROC、FPR、TPR、F1 分数等。

### `sampler.py`

**功能**：实现数据集的采样功能，主要用于减少训练过程中的计算开销。

* **IdentitySampler**：直接返回输入特征，不进行采样操作。
* **BaseSampler**：基础采样器类，提供特征存储和恢复功能，子类需要实现具体的采样方法。
* **GreedyCoresetSampler**：使用贪心算法进行核心集采样，从特征集中选择最有代表性的样本。
* **ApproximateGreedyCoresetSampler**：近似贪心核心集采样方法，减少内存消耗，适用于大数据集。
* **RandomSampler**：随机采样器，从特征集中随机选择指定比例的样本。

### `backbones.py`

**功能**：提供了多个预训练的深度学习模型作为特征提取器（backbone）。

* **load**：
  * 加载指定名称的预训练模型，如 ResNet、VGG、EfficientNet、ViT 等；
  * 额外支持 CLIP 视觉骨干（如 `clip_vit_b16`、`clip_vit_b32`，依赖 `open_clip_torch`）。

### `utils.py`

**功能**：提供各种实用的工具函数。

* **plot_segmentation_images**：生成异常分割结果的可视化图像，并将其保存到指定文件夹。
* **create_storage_folder**：创建存储文件夹，支持迭代模式和覆盖模式。
* **set_torch_device**：设置使用的设备（CPU 或 GPU）。
* **fix_seeds**：固定随机种子，确保结果的可复现性。
* **compute_and_store_final_results**：计算并保存最终的评估结果到 CSV 文件。

## 模块间的依赖关系

1. **`patchcore.py`**：是框架的核心模块，依赖于 `common.py` 提供的工具类（如 `FaissNN` 和 `NearestNeighbourScorer`）进行特征提取、异常检测和评分。`patchcore.py` 还依赖于 `sampler.py` 中的 `GreedyCoresetSampler` 和 `RandomSampler` 等进行特征采样。

2. **`common.py`**：提供了整个框架中常用的工具函数，如特征合并、近邻搜索和特征聚合，多个模块（包括 `patchcore.py` 和 `metrics.py`）都依赖于它。

3. **`metrics.py`**：提供了评估函数，用于计算异常检测的性能指标，主要被 `patchcore.py` 使用来评估图像和像素级的异常检测结果。

4. **`sampler.py`**：提供了数据采样方法，减少训练过程中的冗余数据和内存消耗，主要被 `patchcore.py` 中的训练部分调用。

5. **`backbones.py`**：提供了多个预训练的深度学习模型，主要被 `patchcore.py` 用作特征提取器。

6. **`utils.py`**：提供了辅助功能（如设备设置、图像保存、随机种子固定等），被多个模块调用，确保工作流顺畅和可复现性。
