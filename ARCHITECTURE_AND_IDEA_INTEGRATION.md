# EviMSGT 架构实现态说明（严格按当前代码）

本文档只记录当前仓库中已经实现并可在代码中定位的机制。所有“候选改造”会明确标注为未实现。

## 1. 代码入口与职责

- `scripts/train_single_split.py`
  - 单 split 训练入口（split1..split5）。
  - 调用 `train_5split_ensemble_grid.train_one_split`。
- `scripts/train_5split_ensemble_grid.py`
  - 训练循环、评估指标、checkpoint 保存。
  - 当前最佳模型选择指标是 `val_acc`（不是 `val_auc`）。
- `scripts/train_plat.py`
  - 图构建（SMILES/FASTA）、`BBBP_Dataset`、evidential loss。
- `plat_model/model.py`
  - `GraphTransformer` / `MultiScaleGraphTransformer` / `SubGT` 网络定义。

## 2. 数据到图（当前实现）

在 `scripts/train_plat.py` 中：

- 输入支持 `smiles` 或 FASTA-like `sequence`。
- 节点特征：原子类型、度、formal charge、radical electrons、hybridization、aromatic、H 数。
- 边特征：single/double/triple/aromatic、conjugation、ring。
- 图对象包含：
  - `x`
  - `edge_index`
  - `edge_attr`
  - `atom_residue_index`
  - `pos`（当前为零张量）
  - `atom_z`（原子序数，新增用于拓扑边类型判定）

## 3. MultiScaleGraphTransformer（当前实现）

在 `plat_model/model.py` 的 `MultiScaleGraphTransformer`：

1. 原子级编码：
   - `node_encoder`, `edge_encoder`
   - 多层 `GraphTransformerModule` + `FinalGraphTransformerModule`

2. 残基级编码：
   - 原子到残基池化（mean）
   - 多层 `nn.MultiheadAttention` + FFN + LayerNorm

3. 跨尺度门控写回（已实现 Sigmoid 门控 + 相对几何）：

$$
\Delta p_i = p_i - c_{r(i)}, \quad g_i = [\Delta x_i, \Delta y_i, \Delta z_i, ||\Delta p_i||]
$$

$$
  ext{gate} = \sigma\left(W_g [h_a ; m_r ; g_i]\right)
$$

$$
h_a' = \mathrm{LayerNorm}(h_a + \text{gate} \odot m_r)
$$

其中：
- $h_a$ 为 atom features
- $m_r$ 为 residue message（线性映射后按 `atom_to_residue` 回写）
- $c_{r(i)}$ 为原子 $i$ 所属残基质心
- $g_i$ 为原子相对残基质心几何特征（4 维）

4. 图级读出：

$$
z = [\mathrm{mean}(h_a');\mathrm{max}(h_a');\mathrm{mean}(h_r);\mathrm{max}(h_r)]
$$

5. 输出头：
- 非 evidential：`readout_layer` 输出 2 类 logits
- evidential：`EvidentialClassificationHead`

## 4. 残基层拓扑显式偏置（当前实现）

当前已在 `MultiScaleGraphTransformer` 中实现 residue attention 的拓扑偏置：

1. `residue_dist`：残基图最短路径距离（每个图内 BFS）。
2. `residue_edge_type`：边类型矩阵
   - 0: none
   - 1: residue adjacency（原子边跨残基）
   - 2: disulfide-like（跨残基且端点原子序数均为 16）

3. 偏置构造：

$$
b^{dist}_{ij} = -\lambda \log(1 + d_{ij})
$$

$$
b^{type}_{ij} = \mathrm{Emb}(t_{ij})
$$

$$
b^{topo}_{ij} = b^{dist}_{ij} + b^{type}_{ij}
$$

并通过 `attn_mask` 注入到 `nn.MultiheadAttention`。

实现细节：
- `attn_mask` 为浮点偏置掩码，形状按 `B * num_heads` 扩展后喂入注意力层。
- `key_padding_mask` 继续用于无效 token 屏蔽。

说明：不可达对 `(i,j)` 先以图内可达最大距离 `max_reachable + 1` 进行填充，再参与上式。

## 5. Evidential 头与损失（当前实现）

在 `plat_model/model.py` + `scripts/train_plat.py`：

### 5.1 头部输出

给定 logits $\ell$：

$$
e = \mathrm{softplus}(\ell), \quad \alpha = e + 1
$$

$$
p_k = \frac{\alpha_k}{\sum_j \alpha_j}, \quad u = \frac{K}{\sum_j \alpha_j}
$$

其中 $K$ 为类别数（当前二分类为 2）。

### 5.2 训练损失

`evidential_loss_from_logits` 使用：

- 期望交叉熵项（由 digamma 形式实现）
- KL 到均匀 Dirichlet 先验
- KL 退火系数：

$$
\mathrm{anneal} = \min\left(1, \frac{\mathrm{epoch}+1}{\max(1,\mathrm{anneal\_epochs})}\right)
$$

最终：

$$
\mathcal{L} = \left(\mathrm{NLL}_{Dir} + \mathrm{anneal} \cdot \mathrm{kl\_weight} \cdot \mathrm{KL}\right).\mathrm{mean}()
$$

## 6. 训练与选模（当前实现）

在 `scripts/train_5split_ensemble_grid.py`：

- 优化器：Adam
- 学习率调度：`ReduceLROnPlateau(mode='max')`，监控 `val_acc`
- 保存 best：按 `val_acc` 最大
- 评估指标：AUC/F1/MCC/ACC/BA/SE/SP
- 分类阈值：固定 0.5
- 无 early stopping（训练到设定 epoch）

## 7. 当前未实现（仅候选，不计入实现态）

下列内容目前不在代码中，不应视为已实现：

- ESM-2 / LoRA 注入主干前向
- Prompt-conditioned joint modulation 参数生成（$\gamma,\beta$ 来自任务节点）
- 蒸馏损失项（例如 PLM 与子结构表示对齐）
- PCGrad / GradNorm 等梯度冲突显式处理
- 按 `val_auc` 选模与 early stopping

## 8. 复现实验最小命令（单任务）

```bash
cd /home/shenxin/EviMSGT
/home/shenxin/miniconda3/envs/bbbp-split/bin/python -u scripts/train_single_split.py \
  --csv dataset/fasta_trainval_from_1_5splits.csv \
  --split split1 \
  --epochs 200 \
  --batch_size 32 \
  --model multiscale \
  --use_evidential 1 \
  --mapping_mode helm_force \
  --kl_weight 1e-5 \
  --anneal_epochs 5 \
  --ckpt_dir ckpt/single_split \
  --out_json results/single_train/split1_formal.json
```
