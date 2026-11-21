import json
import time
import numpy as np
import faiss
import pickle

ITEM_EMB_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item_embeddings.npy"
ITEM2IDX_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item2idx.json"
INTER_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\Musical_Instruments.inter.json"
OUT_CANDIDATES_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\topk_candidates.pkl"

TOPK = 50
SEARCH_K = 150 # 多搜一些，为过滤历史留出余量

# ---------- 1. 加载数据和向量 ----------
print("加载嵌入向量...")
embeddings = np.load(ITEM_EMB_PATH).astype(np.float32)
with open(ITEM2IDX_PATH, "r") as f:
    item2idx = json.load(f)
with open(INTER_PATH, "r") as f:
    inter = json.load(f)

idx2item = {v: k for k, v in item2idx.items()}
d = embeddings.shape[1]
print(f"向量维度: {d}, 物品数: {embeddings.shape[0]}")

# ---------- 2. 建立 FAISS 索引 ----------
print("建立 FAISS 索引 (IndexFlatIP)...")
t0 = time.time()
index = faiss.IndexFlatIP(d)  
index.add(embeddings)
build_time = time.time() - t0
print(f"索引建立完成，耗时: {build_time:.4f} 秒")
print(f"索引中文档数量: {index.ntotal}")

# ---------- 3. 构造用户 Query 并检索（Last-Item Query 策略） ----------
print(f"开始为 {len(inter)} 个用户生成 Top-{TOPK} 候选...")
t0 = time.time()

rng = np.random.default_rng(42)
user_candidates = {}

for uid, seq in inter.items():
    # 真正的历史不包含 target
    history_seq = seq[:-1]
    
    if len(history_seq) == 0:
        continue

    # ====== 核心改动：使用最后一个交互物品作为 Query ======
    last_item_idx = history_seq[-1]
    # 取出该物品的向量，并变形成 (1, 1024)
    user_query = embeddings[last_item_idx].reshape(1, -1)
    # 原始向量已归一化，再 normalize 一次确保万无一失
    faiss.normalize_L2(user_query) 
    # ====================================================

    # 搜出 SEARCH_K 个候选
    scores, indices = index.search(user_query, SEARCH_K)
    
    # 过滤集合只包含 history_seq
    history_set = set(history_seq)
    filtered_indices = []
    
    for idx in indices[0]:
        if idx not in history_set and idx != -1: 
            filtered_indices.append(int(idx)) 
        if len(filtered_indices) == TOPK:
            break
            
    # 补齐：随机抽不在历史里、也不重复的物品。
    # 注意：不要用 0 补齐 —— 0 是合法 item id，会制造重复候选。
    while len(filtered_indices) < TOPK:
        cand = int(rng.integers(0, embeddings.shape[0]))
        if cand not in history_set and cand not in filtered_indices:
            filtered_indices.append(cand)
        
    user_candidates[uid] = filtered_indices

search_time = time.time() - t0
print(f"检索完成，总耗时: {search_time:.4f} 秒")
print(f"平均每个用户检索耗时: {search_time / len(inter) * 1000:.2f} ms")

# ---------- 4. 保存结果 ----------
with open(OUT_CANDIDATES_PATH, "wb") as f:
    pickle.dump(user_candidates, f)
print(f"过滤后的 Top-{TOPK} 候选已保存到: {OUT_CANDIDATES_PATH}")

# 打印一个样例
sample_uid = list(user_candidates.keys())[0]
print(f"\n用户 {sample_uid} 的候选前 5 个 item_idx: {user_candidates[sample_uid][:5]}")
