import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\rerank_data.pkl"
ITEM_EMB_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item_embeddings.npy"
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

item_embeddings = np.load(ITEM_EMB_PATH).astype(np.float32)

class RerankDataset(Dataset):
    def __init__(self, data):
        np.random.seed(42)
        np.random.shuffle(data)
        split_idx = int(len(data) * 0.9)
        self.train_data = data[:split_idx]
        self.test_data = data[split_idx:]

    def __len__(self): return len(self.train_data)
    def __getitem__(self, idx):
        item = self.train_data[idx]
        hist_embs = item_embeddings[item['user_history']]
        user_emb = hist_embs.mean(axis=0) 
        cand_embs = item_embeddings[item['cand_indices']]
        return (
            torch.tensor(user_emb, dtype=torch.float32),
            torch.tensor(cand_embs, dtype=torch.float32),
            torch.tensor(item['train_res'], dtype=torch.float32), 
            torch.tensor(item['labels'], dtype=torch.float32)
        )

class ListwiseReranker(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=256):
        super().__init__()
        self.user_proj = nn.Linear(emb_dim, hidden_dim)
        self.cand_proj = nn.Linear(emb_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_emb, cand_embs, res_scores):
        u = self.user_proj(user_emb).unsqueeze(1) 
        c = self.cand_proj(cand_embs)             
        attn_out, _ = self.cross_attn(query=u, key=c, value=c)
        attn_out = attn_out.expand(-1, c.size(1), -1)
        combined = torch.cat([attn_out, c], dim=-1) 
        
        semantic_logits = self.scorer(combined).squeeze(-1)
        
        # ===== 核心修改：严格限制语义的发挥 =====
        # 1. 用 tanh 将语义得分压缩到 [-1, 1]
        # 2. 乘以 0.1，让语义得分最多只能在 SASRec 的基础上微调 ±0.1
        final_logits = res_scores + 0.1 * torch.tanh(semantic_logits)
        
        return final_logits

@torch.no_grad()
def evaluate_reranker(model, test_data):
    model.eval()
    hit_10, ndcg_10, total = 0, 0, 0
    for item in test_data:
        if item['target'] not in item['top50']: continue
        total += 1
        
        hist_embs = item_embeddings[item['user_history']]
        user_emb = hist_embs.mean(axis=0)
        top50_embs = item_embeddings[item['top50']]
        
        u_tensor = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        c_tensor = torch.tensor(top50_embs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        r_tensor = torch.tensor(item['test_res'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        logits = model(u_tensor, c_tensor, r_tensor).squeeze(0) 
        sorted_indices = torch.argsort(logits, descending=True).cpu().numpy()
        sorted_items = [item['top50'][i] for i in sorted_indices]
        
        if item['target'] in sorted_items[:10]:
            hit_10 += 1
            rank = sorted_items.index(item['target']) + 1
            ndcg_10 += 1.0 / np.log2(rank + 1)
            
    print(f"测试用户数: {total}")
    print(f"LEGR 重排 HR@10: {hit_10 / total:.4f}")
    print(f"LEGR 重排 NDCG@10: {ndcg_10 / total:.4f}")

with open(DATA_PATH, "rb") as f: all_data = pickle.load(f)
dataset = RerankDataset(all_data)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ListwiseReranker().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss() 

print("开始训练 Boosting 融合残差模型...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for u, c, r, labels in train_loader:
        u, c, r = u.to(DEVICE), c.to(DEVICE), r.to(DEVICE)
        target_idx = torch.zeros(u.size(0), dtype=torch.long).to(DEVICE) 
        
        logits = model(u, c, r) 
        loss = loss_fn(logits, target_idx)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss / len(train_loader):.4f}")

print("\n" + "="*30 + " 最终评估 " + "="*30)
evaluate_reranker(model, dataset.test_data)
