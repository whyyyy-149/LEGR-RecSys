"""
统一的分桶分析：同一批测试用户、同一套热度定义，比较各召回方法在
tail / mid / head 上的 Recall@50。

用途：支撑"语义的价值集中在长尾"这个结论，且口径完全一致。

用法：python scripts/v2/analyze_buckets.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sid"))

from two_tower import TwoTower, load_data          # noqa: E402
from build_pool import SASRec                      # noqa: E402
from run_v2 import DATA_ROOT, OUT_DIR              # noqa: E402

SASREC_CKPT = os.path.join(REPO_ROOT, "outputs", "SASRec", "best_Musical_Instruments.pt")
SID_DIR = os.path.join(HERE, "sid", "outputs")


def top50_hits_from_scores(score_fn, uids, inter, emb_n, bs=512):
    h50 = np.zeros(len(uids), dtype=np.float32)
    h10 = np.zeros(len(uids), dtype=np.float32)
    for s in range(0, len(uids), bs):
        uu = uids[s:s + bs]
        score = score_fn(uu)                       # (B, N)
        for i, u in enumerate(uu):
            hist = inter[u][:-1]
            score[i, hist] = -1e9
            top = torch.topk(score[i], 50).indices.cpu().numpy()
            tgt = inter[u][-1]
            h50[s + i] = float(tgt in top)
            h10[s + i] = float(tgt in top[:10])
    return h50, h10


def main():
    inter, emb, seq = load_data()
    emb_t = torch.from_numpy(emb)
    emb_n = F.normalize(emb_t, dim=1)
    n_items = emb.shape[0]
    uids = [r["uid"] for r in seq["test"]]
    tg = np.array([inter[u][-1] for u in uids])

    cnt = np.zeros(n_items, dtype=np.float32)
    for s in inter.values():
        for i in s:
            cnt[i] += 1
    pop = np.log1p(cnt)
    q1, q2 = np.percentile(pop[tg], [33, 66])
    buckets = {"tail": (pop[tg] <= q1), "mid": ((pop[tg] > q1) & (pop[tg] <= q2)),
               "head": (pop[tg] > q2)}
    print(f"测试用户 {len(uids)}  tail/mid/head = "
          f"{buckets['tail'].sum()}/{buckets['mid'].sum()}/{buckets['head'].sum()}")

    res = {}

    # 1) 热度
    order = np.argsort(-cnt)
    h50 = np.zeros(len(uids), dtype=np.float32)
    h10 = np.zeros(len(uids), dtype=np.float32)
    for i, u in enumerate(uids):
        hist = set(inter[u][:-1])
        rec = [it for it in order if it not in hist][:50]
        h50[i] = float(inter[u][-1] in rec)
        h10[i] = float(inter[u][-1] in rec[:10])
    res["热度基线"] = (h50, h10)

    # 2) 语义 FAISS（v1 的召回结果，列表按相似度有序）
    with open(os.path.join(DATA_ROOT, "topk_candidates.pkl"), "rb") as f:
        cands = pickle.load(f)
    h50 = np.zeros(len(uids), dtype=np.float32)
    h10 = np.zeros(len(uids), dtype=np.float32)
    for i, u in enumerate(uids):
        c = cands.get(u, [])
        h50[i] = float(inter[u][-1] in set(c))
        h10[i] = float(inter[u][-1] in c[:10])
    res["单 query 语义"] = (h50, h10)

    # 3) SASRec 全库
    sr = SASRec()
    sr.load_state_dict(torch.load(SASREC_CKPT, map_location="cpu"))
    sr.eval()
    I_sr = sr.item_emb.weight.detach()
    I_sr[0] = 0.0

    def score_sasrec(uu):
        seqs = [[x + 1 for x in inter[u][:-1][-50:]] for u in uu]
        L = max(len(x) for x in seqs)
        inp = torch.LongTensor([x + [0] * (L - len(x)) for x in seqs])
        with torch.no_grad():
            urep = sr(inp)
        # 第 0 行是 padding，切掉后列号才等于 item idx
        return (urep @ I_sr.t())[:, 1:]

    res["SASRec"] = top50_hits_from_scores(score_sasrec, uids, inter, emb_n)

    # 4) 双塔
    ck = torch.load(os.path.join(OUT_DIR, "two_tower.pt"), map_location="cpu")
    a = ck["args"]
    tt = TwoTower(emb.shape[1], a["d"], a["pool"])
    tt.load_state_dict(ck["state_dict"])
    tt.eval()
    I_tt = tt.item(emb_t)

    def score_tt(uu):
        h = np.zeros((len(uu), 20), dtype=np.int64)
        m = np.zeros((len(uu), 20), dtype=bool)
        for i, u in enumerate(uu):
            x = inter[u][:-1][-20:]
            h[i, 20 - len(x):] = x
            m[i, 20 - len(x):] = True
        with torch.no_grad():
            uvec = tt.user(emb_t[torch.from_numpy(h)] * torch.from_numpy(m).unsqueeze(-1),
                           torch.from_numpy(m))
            return uvec @ I_tt.t()

    res["双塔"] = top50_hits_from_scores(score_tt, uids, inter, emb_n)

    # 汇总
    for tag, idx in (("Recall@50", 0), ("HR@10", 1)):
        print(f"\n{tag}")
        print(f"{'方法':<16}{'全体':>8}{'tail':>8}{'mid':>8}{'head':>8}")
        for k, (a50, a10) in res.items():
            h = (a50, a10)[idx]
            print(f"{k:<16}{h.mean():>8.4f}" +
                  "".join(f"{h[buckets[b]].mean():>8.4f}" for b in ("tail", "mid", "head")))

    with open(os.path.join(OUT_DIR, "buckets.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"recall50": float(r.mean()), "hr10": float(t.mean()),
                       "recall50_by_bucket": {b: float(r[buckets[b]].mean()) for b in buckets},
                       "hr10_by_bucket": {b: float(t[buckets[b]].mean()) for b in buckets}}
                   for k, (r, t) in res.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
