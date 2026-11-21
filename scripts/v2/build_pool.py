"""
候选池重建 + SASRec 先验分（补上 v1 缺失的基线脚本）

v1 的问题：候选池 = last-item 的 FAISS Top-50，正例天然是"异类"，
一条 -cos(last, cand) 启发式即可取得 HR@10 0.91。

本脚本支持三种池：
  --kind hard    复现 v1（1 正 + 50 难负），仅用于对照/压力测试
  --kind mixed   1 正 + 25 难负 + 25 随机负
  --kind random  1 正 + 100 均匀随机负（文献同口径，推荐主指标）

用法：
  python scripts/v2/build_pool.py --kind random --n_rand 100
输出：
  scripts/v2/outputs/pool_<kind>.pkl  ->  [{uid, pool, prior, tpos, target, hist}]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v2 import DATA_ROOT, REPO_ROOT, _load_pickle  # noqa: E402

HARD_PATH = os.path.join(DATA_ROOT, "topk_candidates.pkl")
INTER_PATH = os.path.join(DATA_ROOT, "musical_instruments.inter.json")
EMB_PATH = os.path.join(DATA_ROOT, "item_embeddings.npy")
SASREC_CKPT = os.environ.get(
    "LEGR_SASREC", os.path.join(REPO_ROOT, "outputs", "SASRec", "best_Musical_Instruments.pt")
)
SEED = 2024

# SASRec 超参需与训练时一致（hidden 64 / 2 block / 2 head / max_len 50）
ITEM_NUM, HIDDEN, MAX_LEN, DROPOUT, N_BLOCK, N_HEAD = 24588, 64, 50, 0.2, 2, 2


class SASRec(nn.Module):
    def __init__(self, item_num=ITEM_NUM, hidden_dim=HIDDEN, max_len=MAX_LEN,
                 dropout=DROPOUT, num_blocks=N_BLOCK, num_heads=N_HEAD):
        super().__init__()
        self.item_emb = nn.Embedding(item_num, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 2, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_blocks)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        emb = self.dropout(self.item_emb(x) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        out = self.encoder(emb, mask=causal, src_key_padding_mask=(x == 0))
        lengths = (x != 0).sum(1)
        return self.norm(out[torch.arange(B, device=x.device), lengths - 1, :])

    def score(self, user_repr, items):
        return (user_repr.unsqueeze(1) * self.item_emb(items)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="random", choices=["hard", "mixed", "random"])
    ap.add_argument("--n_hard", type=int, default=None)
    ap.add_argument("--n_rand", type=int, default=None)
    args = ap.parse_args()

    if args.kind == "random":
        n_hard, n_rand = 0, (args.n_rand or 100)
    elif args.kind == "mixed":
        n_hard, n_rand = (args.n_hard or 25), (args.n_rand or 25)
    else:
        n_hard, n_rand = (args.n_hard or 50), (args.n_rand or 0)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outputs", f"pool_{args.kind}.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"kind={args.kind}  hard={n_hard}  rand={n_rand}  -> {out_path}")

    t0 = time.time()
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        inter = json.load(f)
    hard = _load_pickle(HARD_PATH)
    emb = np.load(EMB_PATH)
    n_items = emb.shape[0]

    model = SASRec()
    model.load_state_dict(torch.load(SASREC_CKPT, map_location="cpu"))
    model.eval()
    print(f"物品 {n_items}，用户 {len(inter)}，SASRec 权重已加载")

    rng = np.random.default_rng(SEED)
    uids = list(inter.keys())
    out = []
    with torch.no_grad():
        for s in range(0, len(uids), 512):
            batch = uids[s:s + 512]
            seqs, metas = [], []
            for u in batch:
                seq = inter[u]
                hist, target = seq[:-1], seq[-1]
                forb = set(hist) | {target}
                if n_hard:
                    hneg = [x for x in hard.get(u, []) if x not in forb][:n_hard]
                    forb |= set(hneg)
                else:
                    hneg = []
                rneg = []
                while len(rneg) < n_rand:
                    c = int(rng.integers(0, n_items))
                    if c not in forb:
                        rneg.append(c)
                        forb.add(c)
                pool = np.array([target] + hneg + rneg, dtype=np.int64)
                seqs.append([x + 1 for x in hist[-MAX_LEN:]])
                metas.append((u, pool, target, hist))

            L = max(len(x) for x in seqs)
            padded = [x + [0] * (L - len(x)) for x in seqs]
            urep = model(torch.LongTensor(padded))
            for bi, (u, pool, target, hist) in enumerate(metas):
                idx = torch.LongTensor(pool[None, :] + 1)
                prior = model.score(urep[bi:bi + 1], idx).squeeze(0).numpy().astype(np.float32)
                out.append({"uid": u, "pool": pool, "prior": prior,
                            "tpos": 0, "target": target, "hist": hist})

            if (s // 512) % 20 == 0:
                print(f"  {s + len(batch)}/{len(uids)}  {time.time() - t0:.0f}s")

    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"保存 {len(out)} 条 -> {out_path}  用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
