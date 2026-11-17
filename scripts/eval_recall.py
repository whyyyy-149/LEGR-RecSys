# eval_recall.py
import json
import pickle

INTER_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\Musical_Instruments.inter.json"
CANDIDATES_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\topk_candidates.pkl"

with open(INTER_PATH, "r") as f:
    inter = json.load(f)
with open(CANDIDATES_PATH, "rb") as f:
    candidates = pickle.load(f)

hit_count = 0
total_users = 0

for uid, seq in inter.items():
    if uid not in candidates or len(seq) < 2:
        continue
    
    total_users += 1
    test_target = seq[-1]
    top50 = candidates[uid]
    
    if test_target in top50:
        hit_count += 1

recall_50 = hit_count / total_users
print(f"统计用户数: {total_users}")
print(f"Top-50 命中数: {hit_count}")
print(f"FAISS 纯语义召回率 (Recall@50): {recall_50:.4f}")