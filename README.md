# LEGR: LLM-Embedding Guided Generative Reranking for Sequential Recommendation

## 1. 项目概述
本项目在 Amazon Reviews 2023 (Musical_Instruments) 数据集上，构建了 **“LLM 语义嵌入 → FAISS 向量召回 → 生成式 Listwise 重排”** 的两阶段推荐链路。
项目旨在探索在生成式架构下，大语言模型（LLM）提取的文本语义特征与序列推荐（SASRec）提取的 ID 协同特征如何有效融合，并在相同候选池下进行了严谨的对比验证。

## 2. 项目结构

LEGR/
├── data/
│   ├── raw/
│   │   ├── benchmark/
│   │   │   └── 5core/
│   │   │       └── last_out_w_his/        # 原始 benchmark 切分数据
│   │   │           ├── Musical_Instruments.train.csv.gz
│   │   │           ├── Musical_Instruments.valid.csv.gz
│   │   │           └── Musical_Instruments.test.csv.gz
│   │   └── meta/                          # 原始 metadata
│   │       └── meta_Musical_Instruments_clean.jsonl
│   └── processed/
│       └── Musical_Instruments/
│           ├── Musical_Instruments.inter.json  # 用户行为序列
│           ├── item2idx.json                   # 物品 ID 映射
│           ├── item_embeddings.npy             # LLM 语义向量 (24587, 1024)
│           ├── topk_candidates.pkl             # FAISS Top-50 召回结果
│           └── rerank_data.pkl                 # 1正50负重排训练数据
├── outputs/
│   └── sasrec/
│       └── best_Musical_Instruments.pt         # SASRec 基线模型权重
├── scripts/
│   ├── build_inter.py           # 构建 inter.json 数据
│   ├── embed_items.py           # 调用 Qwen3 生成语义向量
│   ├── build_faiss.py           # 构建 FAISS 索引并召回 (含历史过滤)
│   ├── eval_recall.py           # 评估召回率 Recall@50
│   ├── build_rerank_data.py     # 构造重排训练集 (1正+50困难负)
│   └── train_reranker.py        # 训练并评估 LEGR (残差 Boosting)
└── README.md

## 3. 数据与评估协议
- **数据规模**：57,439 用户，24,587 物品（5-core，leave-one-out 切分）。
- **评估口径**：严格采用 **同候选池对比（Same-candidate-pool Evaluation）**。为了避免全量评估与召回评估的难度偏差，所有模型（SASRec 与 LEGR）均在 **FAISS 召回的 Top-50 困难负样本池** 下进行 HR@10 / NDCG@10 评估。

## 4. 链路与实验设计

### 4.1 召回阶段 (LLM + FAISS)
- **文本表示**：使用 Qwen3-Embedding-0.6B 提取物品 Title + Description 的 1024 维语义向量，并做 L2 归一化。
- **检索策略**：构建 FAISS `IndexFlatIP` 索引。使用 **Last-Item Query**（用户最近一次交互的语义向量）进行 Top-50 检索，并加入历史交互过滤。
- **基线数据**：纯语义召回率 **Recall@50 = 4.32%**。（证明纯文本语义在缺乏协同信号时的局限性）

### 4.2 重排阶段 (LEGR vs SASRec)
| 模型 | 候选池设置 | HR@10 | NDCG@10 |
| :--- | :--- | :--- | :--- |
| SASRec (判别式) | 1正 + 100随机负 (全局基线) | 0.5148 | 0.3247 |
| **SASRec (同池对齐)** | **FAISS Top-50** | **0.5519** | **0.3308** |
| LEGR (纯语义重排) | FAISS Top-50 | 0.2736 | 0.1409 |
| **LEGR (残差融合)** | **FAISS Top-50** | **0.5472** | **0.3174** |

*注：纯语义重排由于缺乏协同记忆，表现不佳。引入残差融合（将 SASRec 分数作为先验注入 Cross-Attention）后，LEGR 成功逼近了纯协同基线。*

## 5. 核心结论与思考
1. **语义的价值边界**：在纯语义构建的困难候选池中，ID 协同信号（SASRec）占据绝对主导。LLM 语义特征难以在精排阶段实现对协同特征的碾压，其核心价值在于**冷启动兜底与长尾泛化**。
2. **多模态融合的工程妥协**：直接将高维语义特征（1024d）与低维协同分数（1d）拼接会导致特征稀释。本项目采用 **Residual Boosting（残差提升）** 策略，有效防止了语义噪声破坏原有的协同排序，是工业界落地大模型特征的平滑方案。
3. **特征融合边界实证（Alpha 网格搜索）**：在同池对比下，我对残差权重 \(\alpha\) 进行了从 0.0 到 1.0 的网格搜索。实验发现，当 \(\alpha=0.0\) 时 HR@10 最高（0.5519）；随着 \(\alpha\) 增大，指标单调下降。这定量证明了“特征冗余陷阱”：在纯语义召回的候选池中，物品的文本相似度已被穷尽，高维语义特征在精排阶段反而成为噪音。

## 6. 技术栈
- **框架**：PyTorch
- **模型**：Qwen3-Embedding-0.6B, SASRec, Listwise Cross-Attention Reranker
- **工具**：FAISS (IndexFlatIP), Pandas, Numpy
