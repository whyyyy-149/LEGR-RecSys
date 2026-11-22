"""
LEGR v2 —— 修正评估协议后的训练与评估

相对 v1 的改动：
 1) 评估口径：全体测试用户（不再只看召回命中的 4.32%，n 从 246 → 5,744）
 2) 显著性：paired bootstrap 95% CI + McNemar 检验 + 多 seed
 3) 模型：候选间 self-attention（真 listwise）+ 可学习门控融合
 4) 池子：支持 hard / mixed / random 三种候选池，主指标用 1 正 + 100 随机负
 5) 无泄漏：用户向量只取 history（排除 target）
 6) 早停：按用户 80/10/10 切分，用验证集选模型

用法：
  python scripts/v2/build_pool.py --kind random --n_rand 100
  python scripts/v2/run_v2.py --pool random --epochs 3 --seeds 0 1
"""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_ROOT = os.environ.get(
    "LEGR_DATA", os.path.join(REPO_ROOT, "data", "processed", "Musical_Instruments")
)
INTER_PATH = os.path.join(DATA_ROOT, "musical_instruments.inter.json")
EMB_PATH = os.path.join(DATA_ROOT, "item_embeddings.npy")
RERANK_PATH = os.path.join(DATA_ROOT, "rerank_data.pkl")

OUT_DIR = os.environ.get("LEGR_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

TEXT_DIM = 64      # 1024 -> 64 的固定 PCA，避免高维稀释并提速
K = 10
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------- io
def _load_pickle(path):
    """兼容 numpy 2.x 写出的 pickle（numpy._core -> numpy.core）。"""
    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _U(f).load()


def load_basic():
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        inter = json.load(f)
    emb = np.load(EMB_PATH).astype(np.float32)
    return inter, emb


def load_pool(kind):
    """统一成 [{uid, pool(N), prior(N), tpos, target}]"""
    if kind in ("mixed", "random", "hard_v2"):
        path = os.path.join(OUT_DIR, f"pool_{kind}.pkl")
        if not os.path.exists(path):
            raise SystemExit(f"先运行 build_pool.py --kind {kind}")
        return _load_pickle(path)
    raw = _load_pickle(RERANK_PATH)          # v1 原始 51 池
    return [
        {"uid": d["uid"], "pool": np.asarray(d["cand_indices"], dtype=np.int64),
         "prior": np.asarray(d["train_res"], dtype=np.float32), "tpos": 0,
         "target": int(d["target"])}
        for d in raw
    ]


# ---------------------------------------------------------------- features
def fit_pca(X, k, seed=0):
    """随机化 SVD，避免 24587x1024 全量 SVD 的开销。"""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    rng = np.random.default_rng(seed)
    Y = Xc @ rng.normal(size=(X.shape[1], k + 10)).astype(np.float32)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:k].T.astype(np.float32)


def build_features(inter, emb, n_items, tag="v2"):
    cache = os.path.join(OUT_DIR, f"features_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["item_feat"], z["user_feat"], z["uids"]

    mu, V = fit_pca(emb, TEXT_DIM)
    text = ((emb - mu) @ V).astype(np.float32)
    text /= (text.std(0, keepdims=True) + 1e-6)

    cnt = np.zeros(n_items, dtype=np.float32)
    for seq in inter.values():
        for i in seq:
            cnt[i] += 1
    pop = np.log1p(cnt)
    pop = (pop - pop.mean()) / (pop.std() + 1e-6)

    item_feat = np.concatenate(
        [text, pop[:, None], np.ones((n_items, 1), dtype=np.float32)], 1
    )
    np.save(os.path.join(OUT_DIR, "item_pop.npy"), cnt)

    uids = np.array(list(inter.keys()))
    user_feat = np.zeros((len(uids), TEXT_DIM), dtype=np.float32)
    for r, u in enumerate(uids.tolist()):
        h = inter[u][:-1]                      # 必须排除 target
        user_feat[r] = text[h].mean(0) if h else 0.0

    np.savez(cache, item_feat=item_feat, user_feat=user_feat, uids=uids)
    return item_feat, user_feat, uids


# ---------------------------------------------------------------- dataset
class PoolDataset(Dataset):
    """训练时打乱候选顺序，避免"正例永远在第 0 位"被模型利用。
    prior_dropout > 0 时随机丢弃协同先验，逼模型学会在没有 ID 信号时退回语义。"""

    def __init__(self, rows, uid2row, shuffle_pool=False, seed=0, prior_dropout=0.0):
        self.rows = rows
        self.uid2row = uid2row
        self.shuffle_pool = shuffle_pool
        self.prior_dropout = prior_dropout
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = self.rows[i]
        pool, prior, tpos = d["pool"], d["prior"], d["tpos"]
        if self.shuffle_pool:
            p = self.rng.permutation(len(pool))
            pool, prior = pool[p], prior[p]
            tpos = int(np.where(p == tpos)[0][0])
        z = (prior - prior.mean()) / (prior.std() + 1e-6)
        if self.prior_dropout > 0 and self.rng.random() < self.prior_dropout:
            z = np.zeros_like(z)          # 模拟"该物品没有可用的协同信号"
        return (
            torch.from_numpy(np.asarray(pool, dtype=np.int64)),
            torch.from_numpy(np.asarray(z, dtype=np.float32)),
            self.uid2row[d["uid"]],
            int(tpos),
        )


# ---------------------------------------------------------------- models
class ListwiseReranker(nn.Module):
    """真 listwise：候选之间做 self-attention，用户向量作为首 token。
    final = prior_z + gate * tanh(sem)，gate 逐候选、可学习。"""

    def __init__(self, d_in, d=TEXT_DIM, heads=4, layers=2, dropout=0.1,
                 fusion="residual"):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.user_proj = nn.Linear(TEXT_DIM, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.fusion = fusion
        # residual: final = prior + g·tanh(sem)
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)   # 初始 gate≈0.12，先信协同
        # gated: final = a·prior + b·tanh(sem)，(a,b)=softmax(...)，按流行度条件化
        self.gate2 = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 2))
        with torch.no_grad():
            self.gate2[-1].bias.copy_(torch.tensor([1.0, -1.0]))
        self.use_prior = True

    def forward(self, pool_feat, prior_z, user_vec):
        c = self.proj(pool_feat) + self.type_emb[1]
        u = self.user_proj(user_vec).unsqueeze(1) + self.type_emb[0]
        h = self.enc(torch.cat([u, c], 1))[:, 1:, :]
        sem = self.head(h).squeeze(-1)
        tanh_sem = torch.tanh(sem)

        if self.fusion == "gated":
            # 双通道门控：允许模型在冷物品上把权重整体让给语义
            pop = pool_feat[:, :, TEXT_DIM]
            pstd = prior_z.std(1, keepdim=True).expand_as(prior_z)
            w = torch.softmax(
                self.gate2(torch.stack([prior_z, tanh_sem, pop, pstd], -1)), -1)
            base = prior_z if self.use_prior else torch.zeros_like(prior_z)
            return w[..., 0] * base + w[..., 1] * tanh_sem

        g = torch.sigmoid(
            self.gate(torch.stack(
                [prior_z, tanh_sem,
                 prior_z.std(1, keepdim=True).expand_as(prior_z)], -1))
        ).squeeze(-1)
        base = prior_z if self.use_prior else torch.zeros_like(prior_z)
        return base + g * tanh_sem


class LegacyReranker(nn.Module):
    """v1 结构：1 个 query 的 cross-attention，输出广播给所有候选，固定 alpha。
    保留用于同协议对照。"""

    def __init__(self, d_in, d=TEXT_DIM, heads=4, alpha=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.user_proj = nn.Linear(TEXT_DIM, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.scorer = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Linear(128, 1))
        self.alpha = alpha
        self.use_prior = True

    def forward(self, pool_feat, prior_z, user_vec):
        c = self.proj(pool_feat)
        u = self.user_proj(user_vec).unsqueeze(1)
        ctx, _ = self.attn(u, c, c)
        ctx = ctx.expand(-1, c.size(1), -1)      # 广播：所有候选共享同一上下文
        sem = self.scorer(torch.cat([ctx, c], -1)).squeeze(-1)
        base = prior_z if self.use_prior else torch.zeros_like(prior_z)
        return base + self.alpha * torch.tanh(sem)


# ---------------------------------------------------------------- metrics
def target_rank(scores, tpos):
    """ties 用极小随机扰动打散，避免"并列时下标 0 获胜"的伪提升。"""
    jitter = np.random.default_rng(0).normal(0, 1e-6, size=scores.shape).astype(np.float32)
    order = np.argsort(-(scores + jitter), axis=1)
    return np.argmax(order == np.asarray(tpos)[:, None], axis=1) + 1


def hr_ndcg(rank, k=K):
    hit = (rank <= k).astype(np.float32)
    ndcg = np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0).astype(np.float32)
    return hit, ndcg


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    a, b = a.astype(bool), b.astype(bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    z = (abs(n10 - n01) - 1) / math.sqrt(n)
    return n10, n01, float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def line(name, hit, ndcg, base=None):
    lo, hi = boot(hit)
    lo2, hi2 = boot(ndcg)
    print(f"{name:<26} HR@10={hit.mean():.4f} [{lo:.4f},{hi:.4f}]  "
          f"NDCG@10={ndcg.mean():.4f} [{lo2:.4f},{hi2:.4f}]")
    if base is not None:
        d, dl, dh = paired(hit, base)
        w, l, p = mcnemar(hit, base)
        print(f"{'':<26} ΔvsSASRec {d:+.4f} [{dl:+.4f},{dh:+.4f}]  "
              f"McNemar p={p:.2g} (win {w}/lose {l})")


# ---------------------------------------------------------------- train
@torch.no_grad()
def predict(model, ds, item_feat, user_feat, bs=512):
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    sc, tp = [], []
    for pool, pz, urow, tpos in dl:
        out = model(torch.from_numpy(item_feat[pool.numpy()]), pz,
                    torch.from_numpy(user_feat[urow.numpy()]))
        sc.append(out.numpy())
        tp.append(np.asarray(tpos))
    return np.concatenate(sc), np.concatenate(tp)


def train_variant(name, seed, tr, va, te, item_feat, user_feat, args, te_corrupt=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (ListwiseReranker(item_feat.shape[1], fusion=args.fusion)
             if name == "listwise" else LegacyReranker(item_feat.shape[1]))
    model.use_prior = not args.no_prior
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True)

    best = (-1, None, 0)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for pool, pz, urow, tpos in dl:
            logits = model(torch.from_numpy(item_feat[pool.numpy()]), pz,
                           torch.from_numpy(user_feat[urow.numpy()]))
            loss = ce(logits, torch.as_tensor(tpos))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        vs, vt = predict(model, va, item_feat, user_feat)
        vhit, _ = hr_ndcg(target_rank(vs, vt))
        print(f"    ep{ep + 1} loss={tot / len(dl):.4f} valid HR@10={vhit.mean():.4f}")
        if vhit.mean() > best[0]:
            best = (float(vhit.mean()),
                    {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    ts, tt = predict(model, te, item_feat, user_feat)
    hit, ndcg = hr_ndcg(target_rank(ts, tt))
    if te_corrupt is None:
        return hit, ndcg, None, None
    cs, ct = predict(model, te_corrupt, item_feat, user_feat)
    c_hit, c_ndcg = hr_ndcg(target_rank(cs, ct))
    return hit, ndcg, c_hit, c_ndcg


def corrupt_rows(rows, seed=0):
    """把 target 的先验分替换成池内其它候选的分，模拟"该物品在 ID 空间没有表征"。
    用于冷启动的因果实验：看模型在协同信号丢失时能退回语义多少。"""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        prior = np.asarray(r["prior"], dtype=np.float32).copy()
        t = int(r["tpos"])
        j = int(rng.integers(0, len(prior) - 1))
        j = j if j < t else j + 1
        prior[t] = prior[j]
        out.append({**r, "prior": prior})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="random", choices=["hard", "mixed", "random"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", default=["legacy", "listwise"])
    ap.add_argument("--zero", default="", choices=["", "text", "pop"])
    ap.add_argument("--zero_user", action="store_true")
    ap.add_argument("--no_prior", action="store_true")
    ap.add_argument("--prior_dropout", type=float, default=0.0,
                    help="训练时随机丢弃协同先验的概率（冷启动鲁棒性）")
    ap.add_argument("--fusion", default="residual", choices=["residual", "gated"],
                    help="residual: prior+g·sem；gated: 双通道 softmax 门控")
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    args = ap.parse_args()

    t0 = time.time()
    inter, emb = load_basic()
    rows_all = load_pool(args.pool)
    print(f"池类型={args.pool}  用户={len(rows_all)}  物品={emb.shape[0]}")

    item_feat, user_feat, uids = build_features(inter, emb, emb.shape[0])
    uid2row = {u: i for i, u in enumerate(uids.tolist())}
    if args.zero == "text":
        item_feat = item_feat.copy()
        item_feat[:, :TEXT_DIM] = 0.0
        user_feat = user_feat * 0.0
    elif args.zero == "pop":
        item_feat = item_feat.copy()
        item_feat[:, TEXT_DIM] = 0.0
    if args.zero_user:
        user_feat = user_feat * 0.0

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows_all))
    a, b = int(0.8 * len(perm)), int(0.9 * len(perm))
    idx = {"train": perm[:a], "valid": perm[a:b], "test": perm[b:]}
    rows = {k: [rows_all[i] for i in v] for k, v in idx.items()}
    print(f"train/valid/test = {len(rows['train'])}/{len(rows['valid'])}/{len(rows['test'])}")

    tr = PoolDataset(rows["train"], uid2row, True, seed=0, prior_dropout=args.prior_dropout)
    va = PoolDataset(rows["valid"], uid2row, False)
    te = PoolDataset(rows["test"], uid2row, False)
    te_corrupt = PoolDataset(corrupt_rows(rows["test"]), uid2row, False)

    te_rows = rows["test"]
    prior = np.stack([r["prior"] for r in te_rows])
    tpos = np.array([r["tpos"] for r in te_rows])
    p_hit, p_ndcg = hr_ndcg(target_rank(prior, tpos))
    print("\n" + "=" * 92)
    print(f"测试集：全体测试用户，{args.pool} 池（{len(te_rows[0]['pool'])} 候选）")
    print("=" * 92)
    line("SASRec prior", p_hit, p_ndcg)

    pool_np = np.stack([r["pool"] for r in te_rows])
    txt = item_feat[pool_np][:, :, :TEXT_DIM]
    uvec = user_feat[[uid2row[r["uid"]] for r in te_rows]][:, None, :]
    sem = (txt * uvec).sum(-1)
    s_hit, s_ndcg = hr_ndcg(target_rank(sem, tpos))
    line("Text-only (mean hist)", s_hit, s_ndcg, base=p_hit)

    last_idx = np.array([inter[r["uid"]][-2] for r in te_rows])
    anti = -(txt * item_feat[last_idx][:, None, :TEXT_DIM]).sum(-1)
    a_hit, a_ndcg = hr_ndcg(target_rank(anti, tpos))
    line("Anti-last-item (shortcut)", a_hit, a_ndcg, base=p_hit)

    hits, res = {"SASRec prior": p_hit, "Text-only": s_hit}, {"sasrec": float(p_hit.mean())}
    for v in args.variants:
        print(f"\n[{v}]")
        hs, ns, chs = [], [], []
        for sd in args.seeds:
            h, n, ch, cn = train_variant(v, sd, tr, va, te, item_feat, user_feat,
                                         args, te_corrupt)
            print(f"    seed{sd}: HR@10={h.mean():.4f}  (先验被破坏后 {ch.mean():.4f})")
            hs.append(h)
            ns.append(n)
            chs.append(ch)
        h, n = np.mean(hs, 0), np.mean(ns, 0)
        line(f"{v} ({len(args.seeds)}seed avg)", h, n, base=p_hit)
        ch = np.mean(chs, 0)
        print(f"{'':<26} 冷启动探针（破坏 target 的协同先验）: "
              f"HR@10 {h.mean():.4f} -> {ch.mean():.4f}  (SASRec 会退化到 ~0.10)")
        hits[v] = h
        res[v] = {"hr10": float(h.mean()), "ndcg10": float(n.mean()),
                  "hr10_corrupt": float(ch.mean()),
                  "per_seed": [float(x.mean()) for x in hs]}

    pop = np.load(os.path.join(OUT_DIR, "item_pop.npy"))
    tg = np.array([r["target"] for r in te_rows])
    q1, q2 = np.percentile(pop[tg], [33, 66])
    print("\n分桶（按 target 物品交互量，测试集三等分）：")
    print("bucket".ljust(14) + "".join(f"{k:>16}" for k in hits))
    for lo, hi, nm in [(0, q1, "tail"), (q1, q2, "mid"), (q2, 1e9, "head")]:
        m = (pop[tg] >= lo) & (pop[tg] < hi)
        print(f"{nm:<14}" + "".join(f"{v[m].mean():>16.4f}" for v in hits.values())
              + f"  (n={m.sum()})")
    p10 = np.percentile(pop, 10)
    m = pop[tg] <= p10
    print(f"{'cold(p10)':<14}" + "".join(f"{v[m].mean():>16.4f}" for v in hits.values())
          + f"  (n={m.sum()})")

    res.update({"pool": args.pool, "n_test": len(te_rows),
                "prior_dropout": args.prior_dropout})
    suffix = f"_{args.tag}" if args.tag else ""
    with open(os.path.join(OUT_DIR, f"results_{args.pool}{suffix}.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
