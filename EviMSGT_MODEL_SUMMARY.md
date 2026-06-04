# EviMSGT 模型架构与特征总结

本文档根据当前 GitHub 仓库中的源码整理，重点说明 EviMSGT 已经实现的模型结构、输入特征、训练流程、checkpoint 选择逻辑和实验开关。文档只总结当前代码中能够定位到的实现，不额外假设未实现的模块。

## 1. 主要源码文件

| 文件 | 作用 |
|---|---|
| `plat_model/model.py` | 定义核心模型，包括 `GraphTransformer`、`MultiScaleGraphTransformer`、`EvidentialClassificationHead`、`SubGT` 以及消融开关。 |
| `scripts/train_plat.py` | 负责 RDKit 图构建、`BBBP_Dataset`、原子/键特征、残基映射、evidential loss 和旧版训练入口。 |
| `scripts/run_fasta_workflow_batch.py` | 当前 benchmark 批量训练主入口，负责读取 manifest、按 seed 训练、保存 ckpt、做 internal test 和 independent test。 |
| `scripts/summarize_workflow_batch.py` | 汇总 workflow 输出结果，生成 `val_metrics`、`test_metrics`、`independent_metrics`、`independent_test_metrics`。 |
| `scripts/run_representative_search.slurm` | 代表性任务的配置搜索 Slurm 脚本。 |
| `scripts/run_ce_hparam_search.slurm` | CE 版本的超参数搜索 Slurm array 脚本，当前针对任务 6/9/12。 |

## 2. 模型总体定位

当前 EviMSGT 的主模型是 `MultiScaleGraphTransformer`。它不是单纯的序列模型，而是一个多尺度 peptide graph model：

1. 先把多肽转换成原子级分子图。
2. 在原子图上使用 Graph Transformer 编码原子和化学键。
3. 通过 `atom_residue_index` 把原子归属到残基。
4. 将原子特征 mean pooling 到残基层面。
5. 在残基层面使用 self-attention，并可注入 residue topology bias。
6. 将残基上下文通过 cross-scale gate 回写到原子节点。
7. 分别对原子尺度和残基尺度做图级 readout。
8. 最后用普通二分类头或 evidential 分类头输出结果。

因此，当前模型的核心特点是“原子级图建模 + 残基级上下文建模 + 原子-残基跨尺度融合”。

## 3. 输入图与特征

图构建主要在 `scripts/train_plat.py` 中实现。

### 3.1 输入类型

当前代码支持两类输入：

- FASTA-like peptide sequence。
- 可由 RDKit 解析的分子字符串。

在 benchmark workflow 中，输入来自固定 split manifest，例如：

```text
dataset/task_06/split_manifest_task_06.csv
dataset/task_06/independent_manifest_task_06.csv
```

manifest 中常见字段包括：

```text
task, seed, split, sample_id, orig_id, sequence, label
```

运行时通过参数指定 benchmark 数据根目录：

```bash
--manifest_dataset_root /home/shenxin/benchmark/dataset
```

### 3.2 原子节点特征

`NODE_FEATURE_NAMES` 定义了当前原子节点特征：

- atom type one-hot
- degree one-hot
- formal charge
- number of radical electrons
- hybridization one-hot
- aromatic flag
- number of hydrogens one-hot

当前 `MultiScaleGraphTransformer` 在 workflow 中使用：

```python
in_channels = 38
```

也就是说，输入原子节点特征维度为 38。

### 3.3 化学键边特征

`EDGE_FEATURE_NAMES` 定义了当前边特征：

- single bond
- double bond
- triple bond
- aromatic bond
- conjugation
- ring

当前 workflow 中使用：

```python
edge_features = 6
```

### 3.4 残基相关字段

图对象中和多肽残基相关的字段包括：

- `atom_residue_index`：每个原子所属的 residue id。
- `residue_aa_index`：每个残基对应的氨基酸类别 index。
- `atom_z`：原子序数，用于 residue topology 中识别类似二硫键的跨残基连接。
- `pos`：原子坐标，用于 cross-scale geometry gate。

如果图中不存在 `atom_residue_index`，`MultiScaleGraphTransformer` 会退化为“每个原子视作一个 residue”。

### 3.5 坐标模式

坐标由 `build_atom_positions` 构建，受环境变量控制：

```bash
EVIMSGT_POS_MODE
```

当前支持：

| 模式 | 含义 |
|---|---|
| `zero` | 所有原子坐标置零。 |
| `2d` | 使用 RDKit 生成 2D 坐标。 |
| `3d` | 尝试 ETKDG + UFF 优化生成 3D 坐标。 |
| `auto` | 根据图构建逻辑自动选择。 |

图缓存文件名中已经包含 `pos_mode`，因此切换 `zero/2d/3d/auto` 时不会误读旧缓存。

## 4. 原子级 Graph Transformer

原子级编码器由以下部分组成：

- `node_encoder = nn.Linear(in_channels, hidden_dim)`
- `edge_encoder = nn.Linear(edge_features, hidden_dim)`
- 多层 `GraphTransformerModule`
- 最后一层 `FinalGraphTransformerModule`

当前 workflow 构建 `MultiScaleGraphTransformer` 时的典型设置为：

```python
num_hidden_channels = 256
num_layers = 4
num_attention_heads = 4
dropout_rate = <命令行 --dropout>
```

`MultiHeadAttentionLayer` 会根据图的 `edge_index` 在边上进行 attention/message passing。节点特征会被投影为 query/key/value，边特征也会进入 attention/message 计算。

这一部分承担原子级化学图编码，能够利用原子类型、键类型、芳香性、环、共轭等局部化学信息。

## 5. 残基级编码器

残基级路径在原子编码之后执行。

### 5.1 原子到残基 pooling

`_atom_to_residue_pool` 通过 `scatter_add` 实现 mean pooling：

```text
residue_feature[r] = mean(atom_feature[i] for atom i in residue r)
```

因此，残基初始表示来自该残基内部所有原子 embedding 的平均值。

### 5.2 可选残基增强特征

`MultiScaleGraphTransformer` 中实现了三个可选的 residue-level enhancement：

| 开关 | 作用 |
|---|---|
| `use_residue_position` | 添加 learned residue position embedding。 |
| `use_terminal_flags` | 添加 N-terminal / C-terminal flag。 |
| `use_physchem_features` | 添加氨基酸理化性质投影。 |

理化性质表当前包含六类粗粒度属性：

- hydrophobic
- positive
- negative
- polar
- aromatic
- special

这些开关在 `scripts/run_fasta_workflow_batch.py` 中暴露为：

```bash
--use_residue_position
--use_terminal_flags
--use_physchem_features
```

### 5.3 残基 self-attention

残基表示会通过 `to_dense_batch` 转成 batch-first dense tensor，然后进入多层：

- `nn.MultiheadAttention`
- residual connection
- `LayerNorm`
- feed-forward network
- residual connection
- `LayerNorm`

当前默认：

```python
num_residue_layers = 2
```

这一路径用于建模 residue 与 residue 之间的长程关系。

## 6. Residue Topology Bias

当前代码中已经实现 residue-level topology-aware attention bias。

### 6.1 residue graph 构建

`_build_residue_topology` 会根据 atom-level edges 判断跨 residue 的连接，构建：

- `residue_dist`：残基图上的最短路径距离矩阵。
- `residue_edge_type`：残基边类型矩阵。

当前 edge type 定义为：

| edge type | 含义 |
|---|---|
| 0 | 无直接 residue edge。 |
| 1 | 普通跨 residue adjacency。 |
| 2 | disulfide-like bridge，当前通过跨边两端原子序数均为 16 判断。 |

残基最短路径距离按每个 graph 单独计算。

### 6.2 topology attention mask

`_build_topology_attn_mask` 将 residue topology 转换为 attention bias：

```text
topology_bias = -topology_distance_scale * log1p(distance)
                + topology_edge_bias(edge_type)
```

其中：

- `topology_distance_scale` 是可学习参数，初始为 0.3。
- `topology_edge_bias` 是 size 为 3 的 embedding，对应 edge type 0/1/2。

该 bias 会作为 `attn_mask` 注入 residue-level `nn.MultiheadAttention`。

如果使用：

```bash
--ablation_mode no_topology
```

则 residue attention 不使用 topology bias。

## 7. Atom-Residue Cross-Scale Fusion

残基 self-attention 后，模型会将 residue context 回写到 atom node。

对每个原子：

1. 根据 `atom_to_residue` 取对应 residue context。
2. 通过 `atom_from_residue` 将 residue context 投影成 atom message。
3. 计算原子相对所属 residue 质心的几何量：

```text
[dx, dy, dz, ||d||]
```

4. 拼接：

```text
[atom_feature, residue_message, relative_geometry]
```

5. 通过 sigmoid gate：

```text
gate = sigmoid(W[atom_feature, residue_message, relative_geometry])
```

6. 更新 atom feature：

```text
atom_feature' = LayerNorm(atom_feature + gate * residue_message)
```

这个模块是 EviMSGT 当前跨尺度融合的核心。它让 residue-level context 可以影响 atom-level representation，同时保留原子级局部结构。

如果使用：

```bash
--ablation_mode no_geometry
```

则 relative geometry 被置零，但 cross-scale residue message 仍然存在。

## 8. 图级 Readout 与分类头

### 8.1 默认 readout

默认 readout mode 是：

```bash
mean_max
```

模型会分别对 atom scale 和 residue scale 做：

```text
global mean pooling + global max pooling
```

然后拼接：

```text
atom_graph    = [mean(atom_features), max(atom_features)]
residue_graph = [mean(residue_features), max(residue_features)]
graph_repr    = [atom_graph, residue_graph]
```

当 hidden size 为 256 时，最终 graph representation 维度为：

```text
256 * 4 = 1024
```

### 8.2 gated attention readout

代码中也实现了 `GlobalGatedAttentionPool`。当 `readout_mode` 使用 gated attention 路径时，模型会用 gated attention pooling 与 max pooling 组合。

### 8.3 普通 CE 分类头

当关闭 evidential：

```bash
--use_evidential 0
```

模型使用普通二分类 MLP：

```text
Linear(hidden_dim * 4, hidden_dim)
SiLU
Dropout
Linear(hidden_dim, 2)
```

输出为 2 类 logits。

### 8.4 Evidential 分类头

当开启 evidential：

```bash
--use_evidential 1
```

模型使用 `EvidentialClassificationHead`。其逻辑为：

```text
evidence = softplus(logits)
alpha = evidence + 1
probs = alpha / sum(alpha)
uncertainty = num_classes / sum(alpha)
```

默认 forward 仍返回 logits；当 `return_evidential=True` 时，可以返回 probabilities、evidence、Dirichlet alpha 和 uncertainty。

## 9. 消融开关

当前 `MultiScaleGraphTransformer` 支持以下 `ablation_mode`：

| mode | 代码行为 |
|---|---|
| `full` | 完整模型。 |
| `atom_only` | 最终 readout 中 residue context 置零。 |
| `residue_only` | atom readout 前 atom representation 置零。 |
| `no_topology` | residue attention 不使用 topology bias。 |
| `no_geometry` | cross-scale gate 不使用相对几何，geometry vector 置零。 |
| `no_evidential` | 强制关闭 evidential head。 |

命令行示例：

```bash
--ablation_mode full
```

## 10. Loss 设计

当前 workflow 支持 CE 和 evidential 两套训练方式。

### 10.1 Cross Entropy

使用：

```bash
--use_evidential 0
```

时，训练采用 `nn.CrossEntropyLoss`。

类别权重通过：

```bash
--class_weight_mode balanced
```

或：

```bash
--class_weight_mode none
```

控制。`balanced` 会根据训练 split 中每类样本数量计算 class weights。

### 10.2 Evidential Loss

使用：

```bash
--use_evidential 1
```

时，训练调用 `evidential_loss_from_logits`。其核心包括：

- Dirichlet evidence/alpha。
- 分类项。
- 向 uniform Dirichlet 的 KL 正则。
- `--anneal_epochs` 控制 KL annealing。
- `--kl_weight` 控制 KL 权重。

当前 batch workflow 默认：

```bash
--kl_weight 1e-5
--anneal_epochs 5
```

从最近的代表性任务实验看，当前 benchmark setting 下 CE 版本整体更强，因此后续搜索脚本固定使用 CE。

## 11. Benchmark 训练与测试流程

当前主流程是 `scripts/run_fasta_workflow_batch.py`。

一个典型命令为：

```bash
python scripts/run_fasta_workflow_batch.py \
  --manifest_dataset_root /home/shenxin/benchmark/dataset \
  --x_min 6 \
  --x_max 12 \
  --seeds 10,20,30,40,50 \
  --epochs 150 \
  --batch_size 32
```

对每个 task 和 seed，workflow 做以下步骤：

1. 读取 benchmark 固定 split manifest。
2. 按 task 和 seed 过滤 train/val/test。
3. 写出 seed-specific 的 train/val/test CSV。
4. 构建 `BBBP_Dataset` 和图缓存。
5. 训练模型。
6. 按 validation selection metric 捕获主 checkpoint。
7. 额外按 internal test acc 保存一个 internal-test-selected checkpoint。
8. 使用 validation-selected checkpoint 做 independent test。
9. 使用 internal-test-selected checkpoint 再做一版 independent test，结果单独写到 `ind_test_*`。

其中 independent manifest 会在每个 task 下去重并写出一次，不再对每个 seed 重复构建。

## 12. Checkpoint 选择与指标口径

主结果使用 validation-selected checkpoint。

选择指标由：

```bash
--selection_metric
```

控制，当前允许：

- `acc`
- `auc`
- `mcc`
- `ba`
- `f1`
- `combo`

其中：

```text
combo = 0.5 * auc + 0.5 * ba
```

workflow 输出含义如下：

| 输出列 | 含义 |
|---|---|
| `val_*` | validation-selected checkpoint 在 val split 上的指标。 |
| `test_*` | 同一个 validation-selected checkpoint 在 internal test split 上的指标。 |
| `ind_*` | 同一个 validation-selected checkpoint 在 independent set 上的指标。 |
| `ind_test_*` | internal-test-selected checkpoint 在 independent set 上的指标，仅适合作为附表分析。 |

因此，公平主表应使用：

- `val_*`
- `test_*`
- `ind_*`

不应把 `ind_test_*` 或 internal-test-selected checkpoint 当作主结果。

## 13. 当前推荐实验方向

根据最近在代表性任务上的结果，当前较强的配置方向是：

```bash
--use_evidential 0 \
--selection_metric combo \
--class_weight_mode balanced
```

也就是：

- 使用普通 CE loss。
- 按 validation `combo = 0.5 * auc + 0.5 * ba` 选择 checkpoint。
- 使用 balanced class weights。

当前仓库中已经加入 CE 超参数搜索脚本：

```bash
scripts/run_ce_hparam_search.slurm
```

该脚本只跑代表性任务：

```text
6, 9, 12
```

搜索范围为：

```text
learning rate: 1e-4, 3e-4, 5e-4, 1e-3
dropout / weight decay:
  0.05 / 1e-5
  0.10 / 1e-5
  0.20 / 1e-4
  0.10 / 0
```

并固定：

```bash
--use_evidential 0
--selection_metric combo
--class_weight_mode balanced
```

输出目录为：

```text
results/ce_hparam_search/<config_name>
ckpt/ce_hparam_search/<config_name>
```

## 14. 架构特征总结

当前 EviMSGT 已实现的关键特征可以概括为：

- 基于 RDKit 的原子级图构建。
- 38 维原子节点特征。
- 6 维化学键边特征。
- 原子级 Graph Transformer 编码。
- `atom_residue_index` 驱动的 atom-to-residue mapping。
- 原子到残基的 mean pooling。
- 残基级 self-attention。
- residue topology-aware attention bias。
- disulfide-like bridge edge type。
- 可选 residue position embedding。
- 可选 N/C terminal flags。
- 可选 residue physicochemical features。
- 几何感知的 atom-residue cross-scale gate。
- atom/residue 双尺度 readout。
- 普通 CE 分类头。
- 可选 evidential 分类头和 uncertainty 输出。
- 固定 threshold 的二分类评估。
- validation-selected checkpoint 作为主表测试口径。
- internal-test-selected checkpoint 只作为附加分析。

## 15. 当前需要注意的限制

1. 模型依赖 RDKit 图构建和 `atom_residue_index` 的质量。如果残基映射错误，residue-level 模块会受到直接影响。
2. geometry 信号依赖 `EVIMSGT_POS_MODE`。对于只有序列的标准肽任务，2D/3D 生成坐标可能带来增益，也可能引入噪声。
3. `ind_test_*` 来自 internal-test-selected checkpoint，不应作为公平主结果。
4. 当前主比较应统一使用 benchmark 的固定 split manifest。
5. 当前 workflow 没有把 validation threshold search 作为主结果，而是使用固定二分类阈值。
6. evidential head 虽然已经实现，但最近代表性任务结果显示 CE 版本在当前 benchmark 上更稳。

## 16. 推荐运行示例

当前公平 benchmark setting 下，一个代表性命令是：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_fasta_workflow_batch.py \
  --manifest_dataset_root /home/shenxin/benchmark/dataset \
  --x_min 6 \
  --x_max 6 \
  --seeds 10,20,30,40,50 \
  --epochs 150 \
  --batch_size 32 \
  --lr 5e-4 \
  --dropout 0.1 \
  --weight_decay 0 \
  --use_evidential 0 \
  --selection_metric combo \
  --class_weight_mode balanced \
  --pos_mode 2d
```

提交 CE 超参数搜索：

```bash
sbatch scripts/run_ce_hparam_search.slurm
```

