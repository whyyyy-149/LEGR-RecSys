"""
文本双塔召回（DSSM）：把 4.32% 的单 query 语义召回，训成一个可学习的检索模型

对比对象（同一批 5,744 个测试用户、全库 24,587 候选、屏蔽历史）：
    随机                0.20%
    热度基线            7.45%
    纯语义 FAISS        4.32%   ← 本脚本要超过它
    SASRec 全量         8.48%
    生成式 SID          8.86%

结构：
    item 塔：1024d 文本向量 → MLP → 64d → L2 归一化
    user 塔：历史文本向量 → attention pooling → MLP → 64d → L2 归一化
    损失：InfoNCE（in-batch 负例）+ logQ 校正（抵消热门物品被采样的偏差）

用法：
  python scripts/v2/two_tower.py --epochs 5
  python scripts/v2/two_tower.py --epochs 2 --limit 5000      # 冒烟
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_ROOT = os.environ.get(
    "LEGR_DATA", os.path.join(REPO_ROOT, "data", "processed", "Musical_Instruments")
)
INTER_PATH = os.path.join(DATA_ROOT, "musical_instruments.inter.json")
EMB_PATH = os.path.join(DATA_ROOT, "item_embeddings.npy")
SEQ_PATH = os.path.join(HERE, "sid", "outputs", "sid_seq.pkl")
OUT_DIR = os.environ.get("LEGR_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------- 模型
class ItemTower(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class UserTower(nn.Module):
    """历史兴趣的 attention pooling：让模型自己决定看历史里的哪几个物品。
    对比 last-item（只看最后一个）和 mean（等权平均），这是双塔的核心收益来源。"""

    def __init__(self, in_dim=1024, hidden=512, out_dim=64, pool="attn"):
        super().__init__()
        self.pool = pool
        self.q = nn.Parameter(torch.randn(in_dim) / in_dim ** 0.5)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h, mask):
        # h: (B, L, D)  mask: (B, L) bool，True 表示真实位置
        if self.pool == "attn":
            score = (h * self.q).sum(-1)
            score = score.masked_fill(~mask, -1e9)
            w = torch.softmax(score, dim=-1).unsqueeze(-1)
            x = (h * w).sum(1)
        elif self.pool == "mean":
            w = mask.float().unsqueeze(-1)
            x = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        elif self.pool == "last":
            idx = mask.float().cumsum(1).argmax(1)
            x = h[torch.arange(h.size(0)), idx]
        else:
            raise ValueError(self.pool)
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, in_dim=1024, d=64, pool="attn"):
        super().__init__()
        self.user = UserTower(in_dim, 512, d, pool)
        self.item = ItemTower(in_dim, 512, d)

    def forward(self, hist, mask, pos_item):
        return self.user(hist, mask), self.item(pos_item)


# ---------------------------------------------------------------- 数据
def load_data():
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        inter = json.load(f)
    emb = np.load(EMB_PATH).astype(np.float32)
    with open(SEQ_PATH, "rb") as f:
        seq = pickle.load(f)
    return inter, emb, seq


def build_rows(recs, inter, max_hist):
    hist = np.zeros((len(recs), max_hist), dtype=np.int64)
    mask = np.zeros((len(recs), max_hist), dtype=bool)
    tgt = np.zeros(len(recs), dtype=np.int64)
    for i, r in enumerate(recs):
        h = inter[r["uid"]][:-1][-max_hist:]
        hist[i, max_hist - len(h):] = h
        mask[i, max_hist - len(h):] = True
        tgt[i] = r["target"]
    return torch.from_numpy(hist), torch.from_numpy(mask), torch.from_numpy(tgt)


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, emb_t, hist, mask, tgt, bs=512):
    model.eval()
    I = model.item(emb_t)                              # (N, d)
    hits50 = np.zeros(len(tgt), dtype=np.float32)
    hits10 = np.zeros(len(tgt), dtype=np.float32)

    for s in range(0, len(tgt), bs):
        h, m, t = hist[s:s + bs], mask[s:s + bs], tgt[s:s + bs]
        u = model.user(emb_t[h] * m.unsqueeze(-1), m)
        score = u @ I.t()                              # (B, N)
        for i in range(h.size(0)):
            score[i, h[i][m[i]]] = -1e9                # 屏蔽历史
        top = score.topk(50, dim=1).indices
        hits50[s:s + bs] = (top == t[:, None]).any(1).float().numpy()
        hits10[s:s + bs] = (top[:, :10] == t[:, None]).any(1).float().numpy()
    return hits50, hits10


@torch.no_grad()
def baseline_last_item(emb_t, hist, mask, tgt, bs=512):
    """对照：不训练，用最后一个物品的向量检索——应复现 4.32%"""
    hits = np.zeros(len(tgt), dtype=np.float32)
    emb_n = F.normalize(emb_t, dim=1)
    for s in range(0, len(tgt), bs):
        h, m, t = hist[s:s + bs], mask[s:s + bs], tgt[s:s + bs]
        idx = m.float().cumsum(1).argmax(1)
        q = emb_n[h[torch.arange(h.size(0)), idx]]
        score = q @ emb_n.t()
        for i in range(h.size(0)):
            score[i, h[i][m[i]]] = -1e9
        top = score.topk(50, dim=1).indices
        hits[s:s + bs] = (top == t[:, None]).any(1).float().numpy()
    return hits


def ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--max_hist", type=int, default=20)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "last"])
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--no_logq", action="store_true", help="关闭 logQ 校正")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_users_eval", action="store_true",
                    help="负对照：评估时打乱用户向量，指标应大幅下降；不降说明只学了物品侧")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    inter, emb, seq = load_data()
    emb_t = torch.from_numpy(emb)
    tr = seq["train"][:args.limit] if args.limit else seq["train"]
    va, te = seq["valid"], seq["test"]
    print(f"train/valid/test = {len(tr)}/{len(va)}/{len(te)}  物品 {emb.shape[0]}")

    tr_h, tr_m, tr_t = build_rows(tr, inter, args.max_hist)
    va_h, va_m, va_t = build_rows(va, inter, args.max_hist)
    te_h, te_m, te_t = build_rows(te, inter, args.max_hist)

    # logQ：物品作为正例的出现频率（抵消"热门物品被采到"的偏差）
    cnt = np.bincount(tr_t.numpy(), minlength=emb.shape[0]).astype(np.float64) + 1.0
    logq = torch.from_numpy(np.log(cnt / cnt.sum()).astype(np.float32))

    model = TwoTower(emb.shape[1], args.d, args.pool)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    base = baseline_last_item(emb_t, te_h, te_m, te_t)
    print(f"对照（不训练，last-item）Recall@50 = {base.mean():.4f}")

    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_h))
        tot = 0.0
        for s in range(0, len(tr_h), args.bs):
            idx = perm[s:s + args.bs]
            h, m, t = tr_h[idx], tr_m[idx], tr_t[idx]
            u, v = model(emb_t[h] * m.unsqueeze(-1), m, emb_t[t])
            logits = u @ v.t() / args.temp
            if not args.no_logq:
                logits = logits - logq[t].unsqueeze(0)
            loss = F.cross_entropy(logits, torch.arange(idx.size(0)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        vh, v10 = evaluate(model, emb_t, va_h, va_m, va_t)
        print(f"ep{ep+1} loss={tot/(len(tr_h)//args.bs+1):.4f} "
              f"valid R@50={vh.mean():.4f} HR@10={v10.mean():.4f}")
        if vh.mean() > best[0]:
            best = (vh.mean(), {k: v.detach().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    if args.shuffle_users_eval:
        rng = np.random.default_rng(0)
        p = torch.from_numpy(rng.permutation(len(te_h)))
        te_h, te_m = te_h[p], te_m[p]
        print("[负对照] 评估时已打乱用户历史，指标应大幅下降")
    h50, h10 = evaluate(model, emb_t, te_h, te_m, te_t)

    print("\n=== 测试集（全库 24,587，屏蔽历史）===")
    print(f"双塔 Recall@50 = {h50.mean():.4f}  95%CI {ci(h50)}")
    print(f"双塔 HR@10     = {h10.mean():.4f}  95%CI {ci(h10)}")
    print("\n对照：随机 0.20% | 热度 7.45% | 纯语义 4.32% | SASRec 8.48% | 生成式 SID 8.86%")

    pop = np.zeros(emb.shape[0], dtype=np.float32)
    for s in inter.values():
        for i in s:
            pop[i] += 1
    pop = np.log1p(pop)
    tgt_np = te_t.numpy()
    q1, q2 = np.percentile(pop[tgt_np], [33, 66])
    print("\n分桶（按 target 交互量）：")
    for lo, hi, nm in [(0, q1, "tail"), (q1, q2, "mid"), (q2, 1e9, "head")]:
        m = (pop[tgt_np] >= lo) & (pop[tgt_np] < hi)
        print(f"  {nm:<5} R@50={h50[m].mean():.4f}  (n={m.sum()})")

    torch.save({"state_dict": model.state_dict(), "args": vars(args)},
               os.path.join(OUT_DIR, "two_tower.pt"))
    with open(os.path.join(OUT_DIR, "two_tower_report.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "recall50": float(h50.mean()), "recall50_ci": ci(h50),
                   "hr10": float(h10.mean()), "hr10_ci": ci(h10),
                   "elapsed_s": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，模型 -> {OUT_DIR}/two_tower.pt")


if __name__ == "__main__":
    main()
