# v2：评估协议修正与重训

> v1 的 `HR@10 0.5472` **不可复现也不可信**——候选池存在构造性捷径。本目录是修正后的实现。

---

## 1. v1 的三个问题

| # | 问题 | 证据 |
|---|---|---|
| 1 | **候选池有捷径**：池 = last-item 的 FAISS Top-50，正例天然是"异类" | `-cos(last, cand)` 排序 HR@10 = **0.9098** |
| 2 | **模型没在用用户历史**：难负例本身泄露了 query | 用户向量置零：0.7028 → **0.7089**（不降反升） |
| 3 | **条件指标**：只评估召回命中的 4.32% 用户 | n≈246，95% CI ≈ ±6pp，0.5472 vs 0.5519 不显著 |

---

## 2. v2 协议

- 候选池：**1 正 + 100 均匀随机负**（与文献采样评估同口径）；另保留 `hard`（v1 复现）/ `mixed`（25 难负 + 25 随机）作压力测试。
- 评估：**全体测试用户 n=5,744**，HR@10 / NDCG@10。
- 统计：paired bootstrap 2000 次给 95% CI；McNemar 检验；多 seed 取均值。
- 模型：候选间 self-attention（真 listwise）+ 可学习门控 `final = prior_z + g·tanh(sem)`。
- 切分：按用户 80/10/10，验证集早停。
- 泄漏检查：用户向量只取 `history`，排除 target；训练时打乱候选顺序。

---

## 3. 结果（1 正 + 100 随机负，n=5,744）

| 模型 | HR@10 | 95% CI | NDCG@10 | Δ vs SASRec |
|---|---|---|---|---|
| SASRec（ID 协同先验） | 0.5144 | [0.5016, 0.5270] | 0.3287 | — |
| Text-only（历史均值余弦） | 0.3374 | [0.3256, 0.3496] | 0.1995 | −17.7pp |
| LEGR-legacy（v1 结构） | 0.5239 | [0.5107, 0.5366] | 0.3341 | +0.9pp |
| **LEGR-v2（listwise + gate）** | **0.5960** | [0.5837, 0.6078] | **0.3881** | **+8.2pp** [+7.2, +9.0], p<1e-16 |
| LEGR-v2（用户向量置零） | 0.5381 | [0.5251, 0.5512] | 0.3436 | +2.4pp |

随机基线 10/101 ≈ 0.099。

**长尾分桶（按 target 物品交互量三等分）**

| bucket | SASRec | LEGR-v2 | LEGR-v2（无用户） |
|---|---|---|---|
| tail | 0.1323 | **0.2099** | 0.0856 |
| mid | 0.4635 | **0.5903** | 0.4867 |
| head | 0.9204 | **0.9605** | 0.9342 |

结论：
1. 增益不是物品偏置 —— 去掉用户向量掉 5.8pp。
2. 增益集中在长尾 —— tail +7.8pp（相对 +59%）。
3. 无用户时 tail 反而低于基线（0.086 < 0.132）—— 语义信号的价值确实在长尾兜底。

---

## 4. 池子诊断

| 池 | 随机基线 | `-cos(last,cand)` | 用户置零 |
|---|---|---|---|
| hard（v1） | 0.196 | **0.9098** | 0.7089 |
| mixed | 0.196 | 0.2678 | 0.7089 |
| random（v2） | 0.099 | **0.0693** | 0.5381 |

判据：**捷径分低于随机基线才算池子干净**，只有 random 满足。

---

## 5. 运行

```bash
# 1) 建池 + 重算 SASRec 先验（约 20s）
python scripts/v2/build_pool.py --kind random --n_rand 100

# 2) 主实验（约 13 分钟 / 2 seed，CPU）
python scripts/v2/run_v2.py --pool random --epochs 3 --seeds 0 1

# 3) 消融
python scripts/v2/run_v2.py --pool random --epochs 3 --seeds 0 --zero_user
python scripts/v2/run_v2.py --pool hard   --seeds 0 1     # 复现 v1 的问题
```

依赖：`torch`（CPU 可）、`numpy`、`faiss-cpu`（仅建池前需要）。
数据默认读 `data/processed/Musical_Instruments`，可用 `LEGR_DATA` 覆盖；
SASRec 权重默认读 `outputs/SASRec/best_Musical_Instruments.pt`，可用 `LEGR_SASREC` 覆盖。
中间产物写到 `scripts/v2/outputs/`（已被 .gitignore 忽略）。
