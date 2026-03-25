# wat

`wat`（without any train）是一个基于**图像级特征 + KNN 检索**的 AIGC 检测项目。 
使用**双记忆库**（`ai` / `nature`）无需训练额外分类器。

---

## 1. 核心思路（先看这个）

项目的判别逻辑可以概括为三步：

1. 用 backbone 提取图像 embedding（可多层融合）
2. 分别在 `ai` 与 `nature` 记忆库做 KNN 检索，得到两个分数
3. 用分数差做二分类，并用 `ai_generator_conf` 修正；落入不确定区间时输出 `uncertain`

具体定义：

- `raw_margin = score_nature - score_ai`
  - `raw_margin > 0`：更像 AI
  - `raw_margin < 0`：更像 Nature
- `adj_margin = raw_margin * gate`
- `gate = conf_floor + (1-conf_floor) * clip(ai_generator_conf, 0, 1)`

最终分类：

- 若 `|adj_margin| < uncertain_eps` -> `pred_is_ai = -1`（uncertain）
- 否则 `adj_margin > 0` -> `pred_is_ai = 1`（ai）
- 否则 `pred_is_ai = 0`（nature）

---

## 2. 模型与检索流程

### 2.1 双记忆库

- `ai` 库：存放 AI 图像特征，标签为 generator（如 `sdv5`、`midjourney`）
- `nature` 库：存放自然图像特征，标签统一 `nature`

这样做的好处是结构稳定，后续 `ai` 数据集扩展不会直接冲击 `nature` 的标签分布。

### 2.2 分数计算

在 `src/wat/common.py -> NearestNeighbourScorer.predict()` 中，分数按 **纯 L2 距离** 计算：

**输入**
- query 图像 embedding：`q ∈ R^D`
- 记忆库特征：`x_j ∈ R^D`
- top-k：`k = --anomaly_scorer_k`（默认 20）

**步骤 1：特征归一化**

```text
q <- q / ||q||
x_j <- x_j / ||x_j||
```

这样做是为了让不同尺度的特征在同一度量空间里可比较。

**步骤 2：检索 top-k 邻居距离**

```text
d_1, d_2, ..., d_k = KNN(q, memorybank)
```

这里距离越小，表示越相似。

**步骤 3：Top-k 距离求均值**

```text
mean_d = mean(d_1...d_k)
```

**步骤 4：L2 距离映射到分数**

> 前面做了归一化，对单位向量，欧氏距离 d 的范围是 [0, 2]

```text
score = clip(mean_d / 2, 0, 1)
```

解释：
- `mean_d` 越小 -> query 与该库越像
- 因此 `score` 越小 -> 越像该库

最终每张图会得到两组分数：
- `score_ai`：在 AI 库算出来的分数
- `score_nature`：在 Nature 库算出来的分数

### 2.3 AI generator 预测

这部分对应 `src/wat/common.py -> _vote_label()`，只在 **AI 库**上做。

对 top-k 邻居，每个邻居有：
- 距离 `d_i`
- 标签 `g_i`（例如 `sdv5`、`midjourney`）

**步骤 1：把距离变成邻居权重**

```text
w_i = 1 / (d_i + eps)
```

距离越小，`w_i` 越大，近邻影响更大。

**步骤 2：按标签聚合两个量**

```text
weight_sum(g) = Σ w_i, 其中 g_i == g
count(g)      = 邻居中标签 g 出现次数
```

**步骤 3：标签打分（距离 + 数量）**

```text
score(g) = weight_sum(g) * count(g)^p
```

默认 `p=0.5`。 
`p=0` 时只看距离权重；`p` 越大越偏向“数量占优”的标签。

**步骤 4：取赢家标签**

```text
best_label = argmax_g score(g)
```

这就是 `ai_generator`。

**步骤 5：先算基础置信度**

```text
base_conf = score(best_label) / Σ_g score(g)
```

如果一边倒，`base_conf` 接近 1；如果多标签竞争，`base_conf` 会下降。

**步骤 6：再做“杂乱度惩罚”**

为了避免“top-k 很杂但被硬判高置信”，会做两种惩罚：

```text
entropy_norm = H(score分布) / log(n_unique)        # 分布越均匀越杂
unique_ratio = (n_unique - 1) / (k - 1)            # 不同标签越多越杂
penalty = (1 - αe*entropy_norm) * (1 - αu*unique_ratio)
conf = clip(base_conf * penalty, 0, 1)
```

默认 `αe=0.6`、`αu=0.4`。

输出：
- `ai_generator = best_label`
- `ai_generator_conf = conf`
- `ai_generator_conf` 越低，说明 generator 归属越不稳定。

### 2.4 最终二分类规则

对应 `bin/run_wat.py` 的推理主循环，逻辑是“先证据差，再置信度门控，再不确定筛除”。

**步骤 1：基础 margin**

```text
raw_margin = score_nature - score_ai
```

- `raw_margin > 0`：偏 AI
- `raw_margin < 0`：偏 Nature

**步骤 2：用 generator 置信度做门控**

```text
gate = conf_floor + (1 - conf_floor) * clip(ai_generator_conf, 0, 1)
adj_margin = raw_margin * gate
```

默认 `conf_floor=0.35`，因此：
- `ai_generator_conf=1` -> `gate=1`（不衰减）
- `ai_generator_conf=0` -> `gate=0.35`（只保留 35% 的 margin）

这一步的作用是： 
当 AI 归属本身不稳定（`ai_generator_conf` 低）时，把分类证据拉回到更保守的范围。

**步骤 3：不确定区间判定**

```text
if |adj_margin| < uncertain_eps:
    pred_is_ai = -1   # uncertain
elif adj_margin > 0:
    pred_is_ai = 1    # ai
else:
    pred_is_ai = 0    # nature
```

默认 `uncertain_eps=0.01`。 
`eps` 越大，系统越保守，uncertain 比例越高，误报通常会下降但覆盖率会下降。

**步骤 4：指标统计口径**

- `acc(certain)`：只在 `pred_is_ai != -1` 的样本上统计
- `coverage = certain样本数 / 总样本数`
- `uncertain_rate = 1 - coverage`

---

## 3. 代码结构与职责

- `bin/run_wat.py`
  - 主入口（`train` / `infer` / `both`）
  - 负责：数据装配、双库构建/加载、评估、CSV 导出
  - 包含 margin+置信度融合与 uncertain 判定

- `src/wat/wat.py`
  - WAT 主类：`fit()`、`predict()`、`predict_with_meta()`
  - 负责 memorybank 填充、推理输出、元信息回传

- `src/wat/common.py`
  - KNN 后端（`BruteNN` / `FaissNN`）
  - `NearestNeighbourScorer`（分数 + generator 投票逻辑）

- `src/wat/backbones.py`
  - backbone 统一加载（含 open_clip）

- `src/wat/sampler.py`
  - 记忆库采样策略（`random` / `pca` / `central` 等）

- `src/wat/datasets/tiny_genimage.py`
  - Tiny-GenImage 风格数据读取
  - 输出 `generator`、`dataset_name`、`is_ai` 等字段

---

## 4. 数据组织约定

推荐目录：

```text
data_path/
  <dataset_name>/
    train/
      ai/...
      nature/...
    val/ 或 test/
      ai/...
      nature/...
```

也支持：

```text
.../split/ai/<generator>/*.png
```

若 `ai` 下无 generator 子目录，默认使用 `<dataset_name>` 作为 generator 标签。

---

## 5. 快速开始

先设置：

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

训练记忆库：

```bash
python ./bin/run_wat.py \
  --phase train \
  --data_path ~/datasets/tiny_genimage \
  --dataset_names sdv5 glide midjourney \
  --banknames ai nature \
  --pc_save_root ~/memorybanks_wat
```

仅推理：

```bash
python ./bin/run_wat.py \
  --phase infer \
  --data_path ~/datasets/tiny_genimage \
  --test_dataset_names sdv5 \
  --banknames ai nature \
  --pc_save_root ~/memorybanks_wat
```

训练 + 推理：

```bash
python ./bin/run_wat.py \
  --phase both \
  --data_path ~/datasets/tiny_genimage \
  --dataset_names sdv5 \
  --test_dataset_names sdv5
```

---

## 6. 关键参数

- `--layers_to_extract_from`：多层特征配置（影响表征能力）
- `--anomaly_scorer_k`：KNN 的 top-k
- `--sampler_name` / `--coreset_percentage`：记忆库容量与速度平衡
- `--nn_method`：`auto | flat | ivfpq`（faiss 可用时生效）
- `--ai_conf_floor`：置信度融合下限（越小越依赖 `ai_generator_conf`）
- `--uncertain_eps`：不确定区间阈值（越大越保守）

---

## 7. 输出与评估

### 7.1 终端指标

当前代码（`bin/run_wat.py` + `src/wat/eval_tools.py`）会打印三层指标：

1) 数据集级（Per-dataset）  
2) 全局级（Overall）  
3) 全局按 generator 分组（Overall per generator）

下面是每个指标的**详细口径**（名称与代码保持一致）：

说明：公式里的代码变量 `label_is_ai / pred_is_ai`，在 `test_scores.csv` 中分别对应
`ground_truth_is_ai / predicted_is_ai_with_uncertainty`。

- `classification_accuracy_with_uncertainty`
  - 含义：二分类总体准确率（`ai` vs `nature`），把 `uncertain` 当作错误。
  - 公式：`mean(pred_is_ai == label_is_ai)`，其中 `pred_is_ai in {1,0,-1}`，`label_is_ai in {1,0}`。
  - 注意：因为 `-1` 永远不等于真值标签 `0/1`，所以会拉低该值。

- `classification_accuracy_on_certain_samples`
  - 含义：仅在“模型给出确定判断”的样本上统计二分类准确率。
  - 过滤：`pred_is_ai != -1`。
  - 公式：`mean(pred_is_ai == label_is_ai | pred_is_ai != -1)`。

- `classification_certain_sample_coverage`
  - 含义：模型给出确定判断（非 uncertain）的覆盖率。
  - 公式：`mean(pred_is_ai != -1)`。

- `classification_uncertainty_rate`
  - 含义：不确定率。
  - 公式：`1 - classification_certain_sample_coverage`。

- `ai_detection_accuracy_with_uncertainty`
  - 含义：在真实 AI 样本上，“是否判为 AI”的准确率；`uncertain` 记错。
  - 样本子集：`label_is_ai == 1`。
  - 正确条件：`pred_is_ai == 1`。
  - 公式：`mean(pred_is_ai == 1 | label_is_ai == 1)`。

- `ai_detection_accuracy_on_certain_samples`
  - 含义：在真实 AI 且 certain 的子集上，“是否判为 AI”的准确率。
  - 样本子集：`label_is_ai == 1` 且 `pred_is_ai != -1`。
  - 公式：`mean(pred_is_ai == 1 | label_is_ai == 1, pred_is_ai != -1)`。

- `ai_detection_certain_sample_coverage`
  - 含义：真实 AI 样本中，被模型“确定给出 ai/nature”判断的覆盖率。
  - 样本子集：`label_is_ai == 1`。
  - 公式：`count(label_is_ai == 1 and pred_is_ai != -1) / count(label_is_ai == 1)`。

配套样本计数字段（日志和 CSV 中会出现）：
- `total_sample_count`
- `certain_prediction_sample_count`
- `true_ai_sample_count`
- `true_ai_certain_prediction_sample_count`
- `true_ai_predicted_as_ai_sample_count`

### 7.2 CSV 输出文件

当前会输出 5 个 CSV：

1) `runs/test_scores.csv`（逐样本明细）  
完整列名（按顺序）：
`dataset_name,image_path,anomaly_score_from_ai_bank,anomaly_score_from_nature_bank,relative_difference_to_nature_score,symmetric_score_difference,ground_truth_is_ai,predicted_is_ai_with_uncertainty,predicted_label_text,uncertainty_flag,raw_margin_nature_minus_ai,confidence_adjusted_margin_nature_minus_ai,ai_confidence_gate_weight,predicted_generator_for_final_label,predicted_generator_from_ai_bank,predicted_generator_confidence_from_ai_bank,predicted_generator_base_confidence_before_diversity_penalty,predicted_generator_diversity_penalty,topk_unique_label_count,topk_entropy_normalized,topk_unique_ratio,predicted_label_from_nature_bank,ground_truth_generator_name,ground_truth_dataset_name`

2) `runs/per_dataset_ai_evaluation_summary.csv`（按数据集汇总）  
完整列名（按顺序）：
`dataset_name,total_sample_count,certain_prediction_sample_count,true_ai_sample_count,true_ai_certain_prediction_sample_count,true_ai_predicted_as_ai_sample_count,classification_accuracy_with_uncertainty,classification_accuracy_on_certain_samples,classification_certain_sample_coverage,classification_uncertainty_rate,ai_detection_accuracy_with_uncertainty,ai_detection_accuracy_on_certain_samples,ai_detection_certain_sample_coverage`

3) `runs/per_dataset_per_ai_generator_evaluation_summary.csv`（按数据集、按 generator 汇总）  
完整列名（按顺序）：
`dataset_name,ground_truth_ai_generator_name,true_ai_sample_count,true_ai_certain_prediction_sample_count,true_ai_predicted_as_ai_sample_count,ai_detection_accuracy_with_uncertainty,ai_detection_accuracy_on_certain_samples,ai_detection_certain_sample_coverage`

4) `runs/overall_ai_evaluation_summary.csv`（全局单行汇总）  
完整列名（按顺序）：
`classification_accuracy_with_uncertainty,classification_accuracy_on_certain_samples,classification_certain_sample_coverage,classification_uncertainty_rate,ai_detection_accuracy_with_uncertainty,ai_detection_accuracy_on_certain_samples,ai_detection_certain_sample_coverage,true_ai_sample_count,true_ai_certain_prediction_sample_count,true_ai_predicted_as_ai_sample_count`

5) `runs/overall_per_ai_generator_evaluation_summary.csv`（全局按 generator 汇总）  
完整列名（按顺序）：
`ground_truth_ai_generator_name,true_ai_sample_count,true_ai_certain_prediction_sample_count,true_ai_predicted_as_ai_sample_count,ai_detection_accuracy_with_uncertainty,ai_detection_accuracy_on_certain_samples,ai_detection_certain_sample_coverage`

---

## 8. 注意事项

- 修改 generator 标注规则后，必须重新 `--phase train` 重建 memorybank
- 使用 CLIP backbone 时，确认 open_clip 的 pretrained 配置有效
- 无 faiss 依赖时可直接用 BruteNN，功能完整但大规模检索更慢

---

## 9. 推理流程图

```text
                +---------------------------+
                | 输入图像 x                |
                +------------+--------------+
                             |
                             v
                +---------------------------+
                | backbone 提特征 + 聚合    |
                | 得到 image embedding      |
                +------------+--------------+
                             |
              +--------------+---------------+
              |                              |
              v                              v
   +-----------------------+      +-----------------------+
   | 在 AI memorybank 检索 |      | 在 Nature memorybank检索 |
   | top-k -> score_ai     |      | top-k -> score_nature  |
   +-----------+-----------+      +-----------+-----------+
               |                              |
               +--------------+---------------+
                              |
                              v
                +---------------------------+
                | raw_margin =              |
                | score_nature - score_ai   |
                +------------+--------------+
                             |
                             v
                +---------------------------+
                | AI库邻居投票得到          |
                | ai_generator, ai_conf     |
                +------------+--------------+
                             |
                             v
                +---------------------------+
                | gate = floor + (1-floor)* |
                | clip(ai_conf,0,1)         |
                | adj_margin = raw*gate      |
                +------------+--------------+
                             |
                             v
                +----------------------------+
                | if |adj_margin| < eps      |
                |    -> uncertain (-1)       |
                | elif adj_margin > 0        |
                |    -> ai (1)               |
                | else                       |
                |    -> nature (0)           |
                +------------+---------------+
                             |
                             v
                +----------------------------+
                | 输出 pred_is_ai / label /  |
                | generator / conf / CSV     |
                +----------------------------+
```
