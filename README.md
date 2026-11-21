# LEGR: LLM-Embedding Guided Generative Reranking for Sequential Recommendation

> **⚠️ 实验协议修正（2026-11）**
> v1 的指标建立在有缺陷的候选池上，**下方 4.2 节的数字不可作为结论引用**。
> 修正后的协议、结果与代码见 [`scripts/v2/README.md`](scripts/v2/README.md)。
> 一句话：v1 的候选池是「last item 的 FAISS Top-50」，正例天然是其中唯一"没被选进来"的异类，
> 一条 `-cos(last, cand)` 启发式即可取得 HR@10 **0.9098**；换成 1 正 + 100 随机负后，
> 重训的 Listwise 重排器 HR@10 从 **0.5144 → 0.5960（+8.2pp）**，长尾桶 0.132 → 0.210。

## 1. 项目概述
本项目在 Amazon Reviews 2023 (Musical_Instruments) 数据集上，构建了 **“LLM 语义嵌入 → FAISS 向量召回 → 生成式 Listwise 重排”** 的两阶段推荐链路。
项目旨在探索在生成式架构下，大语言模型（LLM）提取的文本语义特征与序列推荐（SASRec）提取的 ID 协同特征如何有效融合，并在相同候选池下进行了严谨的对比验证。

## 2. 项目结构

```text
LEGR/
├── data/
│   ├── raw/
│   │   ├── benchmark/5core/last_out_w_his/  # 原始 benchmark 切分数据 (.csv.gz)
│   │   └── meta/                            # 原始 metadata (.jsonl)
│   └── processed/Musical_Instruments/
│       ├── Musical_Instruments.inter.json   # 用户行为序列
│       ├── item2idx.json                    # 物品 ID 映射
│       ├── item_embeddings.npy              # LLM 语义向量 (24587, 1024)
│       ├── topk_candidates.pkl              # FAISS Top-50 召回结果
│       └── rerank_data.pkl                  # 1正+50负重排训练数据
├── outputs/
│   └── sasrec/
│       └── best_Musical_Instruments.pt      # SASRec 基线模型权重
├── scripts/
│   ├── build_inter.py                       # 构建 inter.json 数据
│   ├── embed_items.py                       # 调用 Qwen3 生成语义向量
│   ├── build_faiss.py                       # 构建 FAISS 索引并召回
│   ├── eval_recall.py                       # 评估召回率 Recall@50
│   ├── build_rerank_data.py                 # [v1] 构造重排训练集 (1正+50困难负，池有缺陷)
│   ├── train_reranker.py                    # [v1] 训练并评估 LEGR (广播式 attention + 固定 α)
│   └── v2/                                  # ★ 修正评估协议后的实现
│       ├── build_pool.py                    # 候选池构建 (hard/mixed/random) + SASRec 先验
│       ├── run_v2.py                        # 训练 + 全体用户评估 + bootstrap/McNemar
│       └── README.md                        # v1 问题、v2 结果、诊断
└── README.md
```

## 3. 数据与评估协议
- **数据规模**：57,439 用户，24,587 物品（5-core，leave-one-out 切分）。
- **评估口径**：严格采用 **同候选池对比（Same-candidate-pool Evaluation）**。为了避免全量评估与召回评估的难度偏差，所有模型（SASRec 与 LEGR）均在 **FAISS 召回的 Top-50 困难负样本池** 下进行 HR@10 / NDCG@10 评估。

## 4. 链路与实验设计

### 4.1 召回阶段 (LLM + FAISS)
- **文本表示**：使用 Qwen3-Embedding-0.6B 提取物品 Title + Description 的 1024 维语义向量，并做 L2 归一化。
- **检索策略**：构建 FAISS `IndexFlatIP` 索引。使用 **Last-Item Query**（用户最近一次交互的语义向量）进行 Top-50 检索，并加入历史交互过滤。
- **基线数据**：纯语义召回率 **Recall@50 = 4.32%**。（证明纯文本语义在缺乏协同信号时的局限性）

### 4.2 重排阶段 (LEGR vs SASRec)

> ⚠️ **本节数字测量于有缺陷的候选池，仅作 v1 历史记录保留。**
> 修正协议后的结果：SASRec 0.5144 → LEGR-v2 **0.5960**（+8.2pp, 95%CI [+7.2,+9.0]），详见 [`scripts/v2/README.md`](scripts/v2/README.md)。

| 模型 | 候选池设置 | HR@10 | NDCG@10 |
| :--- | :--- | :--- | :--- |
| SASRec (判别式) | 1正 + 100随机负 (全局基线) | 0.5148 | 0.3247 |
| **SASRec (同池对齐)** | **FAISS Top-50** | **0.5519** | **0.3308** |
| LEGR (纯语义重排) | FAISS Top-50 | 0.2736 | 0.1409 |
| **LEGR (残差融合)** | **FAISS Top-50** | **0.5472** | **0.3174** |

*注：该表评估的池子由 last-item 近邻构成，正例是池中唯一"非近邻"项，存在构造捷径（`-cos(last,cand)` 排序可达 0.9098）。上表数字不可与文献或 v2 结果直接比较。*

## 5. 核心结论与思考

1. **语义的价值边界**：在纯语义构建的困难候选池中，ID 协同信号（SASRec）占据绝对主导。LLM 语义特征难以在精排阶段实现对协同特征的碾压，其核心价值在于**冷启动兜底与长尾泛化**。

2. **多模态融合的工程妥协**：直接将高维语义特征（1024d）与低维协同分数（1d）拼接会导致特征稀释。本项目采用 **Residual Boosting（残差提升）** 策略，有效防止了语义噪声破坏原有的协同排序，是工业界落地大模型特征的平滑方案。

3. **特征融合边界实证（Alpha 网格搜索）**：在同池对比下，我对残差权重 \(\alpha\) 进行了从 0.0 到 1.0 的网格搜索。

    | Alpha (\(\alpha\)) | HR@10 | NDCG@10 | 现象分析 |
    | :--- | :--- | :--- | :--- |
    | **0.0** | **0.5519** | **0.3308** | 协同信号在语义候选池中占据绝对主导 |
    | 0.1 | 0.5472 | 0.3162 | 语义作为微小扰动，基本保持基线水平 |
    | 0.2 | 0.5236 | 0.2865 | 语义噪声开始破坏协同排序 |
    | 0.5 | 0.4717 | 0.2294 | 高维语义严重稀释低维协同 |
    | 1.0 | 0.4528 | 0.2209 | 纯语义重排，指标大幅暴跌 |

    实验发现，当 \(\alpha=0.0\) 时 HR@10 最高（0.5519）；随着 \(\alpha\) 增大，指标单调下降。这定量证明了**“特征冗余陷阱”**：在纯语义召回的候选池中，物品的文本相似度已被穷尽，高维语义特征在精排阶段反而成为噪音。

## 6. 技术栈
- **框架**：PyTorch
- **模型**：Qwen3-Embedding-0.6B, SASRec, Listwise Self-Attention Reranker
- **工具**：FAISS (IndexFlatIP), Pandas, Numpy

---

## 7. v2：评估协议修正与重训（2026-11）

v1 的结论建立在三个未经检验的假设上，修正后结论反转。

### 7.1 v1 的三个问题

| # | 问题 | 证据 |
|---|---|---|
| 1 | 候选池 = last-item 的 FAISS Top-50，正例天然是池中唯一的"异类" | `-cos(last, cand)` 排序 HR@10 = **0.9098** |
| 2 | 难负例本身泄露了用户 query，模型并未使用用户历史 | 用户向量置零：0.7028 → **0.7089**（不降反升） |
| 3 | 指标是条件指标，只在 4.32% 召回命中的用户上统计 | n≈246，95% CI ≈ ±6pp，原差异不显著 |

### 7.2 v2 协议

- 候选池：**1 正 + 100 均匀随机负**（文献采样评估同口径）；`hard` / `mixed` 作为压力测试保留。
- 评估：全体测试用户 **n=5,744**，HR@10 / NDCG@10。
- 统计：paired bootstrap 95% CI + McNemar 检验 + 多 seed。
- 模型：候选间 self-attention（真 listwise） + 可学习门控 `final = prior_z + g·tanh(sem)`。
- 防泄漏：用户向量只取 history（排除 target）；训练时打乱候选顺序；按用户 80/10/10 切分并早停。

### 7.3 v2 结果（1 正 + 100 随机负，n=5,744）

| 模型 | HR@10 | 95% CI | NDCG@10 | Δ vs SASRec |
|---|---|---|---|---|
| SASRec（ID 协同先验） | 0.5144 | [0.5016, 0.5270] | 0.3287 | — |
| Text-only（历史均值余弦） | 0.3374 | [0.3256, 0.3496] | 0.1995 | −17.7pp |
| LEGR-legacy（v1 结构） | 0.5239 | [0.5107, 0.5366] | 0.3341 | +0.9pp |
| **LEGR-v2（listwise + gate）** | **0.5960** | [0.5837, 0.6078] | **0.3881** | **+8.2pp**, p<1e-16 |
| LEGR-v2（用户向量置零） | 0.5381 | [0.5251, 0.5512] | 0.3436 | +2.4pp |

长尾分桶（按 target 物品交互量三等分）：tail **0.132 → 0.210**，mid 0.464 → 0.590，head 0.920 → 0.961；
用户向量置零后 tail 掉到 **0.086**（低于基线）。语义特征的价值集中在长尾，而非头部。

```bash
python scripts/v2/build_pool.py --kind random --n_rand 100
python scripts/v2/run_v2.py --pool random --epochs 3 --seeds 0 1
```

完整说明见 [`scripts/v2/README.md`](scripts/v2/README.md)。
