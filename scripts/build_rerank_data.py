import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class SASRec(nn.Module):
    def __init__(self, item_num, hidden_dim, max_len, dropout, num_blocks, num_heads):
        super().__init__()
        self.item_emb = nn.Embedding(item_num, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_blocks)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        emb = self.dropout(self.item_emb(x) + self.pos_emb(pos))
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        out = self.encoder(emb, mask=causal_mask, src_key_padding_mask=(x == 0))
        lengths = (x != 0).sum(dim=1)
        return self.norm(out[torch.arange(B, device=x.device), lengths - 1, :])
    def score(self, user_repr, items):
        return (user_repr.unsqueeze(1) * self.item_emb(items)).sum(-1)

INTER_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\Musical_Instruments.inter.json"
CANDIDATES_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\topk_candidates.pkl"
OUT_DATA_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\rerank_data.pkl"
SASREC_PATH = r"C:\Users\wuhan\SASRec.pytorch\best_Musical_Instruments.pt"

with open(INTER_PATH, "r") as f: inter = json.load(f)
with open(CANDIDATES_PATH, "rb") as f: candidates = pickle.load(f)

device = torch.device("cpu")
sasrec = SASRec(24588, 64, 50, 0.2, 2, 2).to(device)
sasrec.load_state_dict(torch.load(SASREC_PATH, map_location=device))
sasrec.eval()

def min_max_norm(x):
    # 将特征缩放到 [0, 1] 范围，防止梯度爆炸
    return (x - x.min()) / (x.max() - x.min() + 1e-5)

def build_dataset(user_dict, cand_dict):
    data = []
    with torch.no_grad():
        for uid, seq in tqdm(user_dict.items(), desc="构造样本"):
            if len(seq) < 2: continue
            history_seq = seq[:-1]
            target = seq[-1]
            top50 = cand_dict.get(uid, [])
            
            inp = [x + 1 for x in history_seq[-50:]]
            user_repr = sasrec(torch.LongTensor([inp]).to(device))
            
            # 1. 构造训练用的候选池 (1 正 + 50 负)
            cand_pool = [target]
            for item in top50:
                if item != target: cand_pool.append(item)
                if len(cand_pool) == 51: break
            while len(cand_pool) < 51: cand_pool.append(0)
            
            # 计算训练残差并归一化
            train_res = sasrec.score(user_repr, torch.LongTensor([[x + 1 for x in cand_pool]]).to(device)).squeeze(0).cpu().numpy()
            train_res = min_max_norm(train_res)
            
            # 2. 计算测试专用的残差 (只用原始 top50)
            test_res = sasrec.score(user_repr, torch.LongTensor([[x + 1 for x in top50]]).to(device)).squeeze(0).cpu().numpy()
            test_res = min_max_norm(test_res)
            
            labels = [1.0] + [0.0] * 50
            
            data.append({
                'uid': uid,
                'user_history': history_seq,
                'cand_indices': cand_pool,      
                'train_res': train_res,     # 训练特征 (51,)
                'test_res': test_res,       # 测试特征 (50,)
                'labels': labels,               
                'target': target,
                'top50': top50             
            })
    return data

dataset = build_dataset(inter, candidates)
with open(OUT_DATA_PATH, "wb") as f: pickle.dump(dataset, f)
print(f"共生成 {len(dataset)} 个训练样本。")