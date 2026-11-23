"""
端到端评估：双塔召回 Top-50 → listwise 重排 → 全库 HR@10

和之前的区别：
  - 之前：所有模型在"人为构造的候选池"上比排序能力（1 正 + 100 随机负）
  - 现在：候选池 = 双塔真实召回结果，target 只在被召回时才存在（不做任何注入）
    这样算出来的 HR@10 就是线上能拿到的效果，天花板是召回率。

流程：
  1. 双塔给所有用户打分，取 Top-50（屏蔽历史）
  2. 训练池：[target] + Top-50（去掉 target 自身）—— 训练需要一个正例
  3. 验证/测试池：纯 Top-50，不注入 target
  4. 重排器：final = prior_z + g·tanh(sem)，prior 用双塔分数
  5. 指标：召回率 / 条件 HR@10 / 端到端 HR@10

用法：
  python scripts/v2/end_to_end.py --epochs 4 --seeds 0 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from two_tower import TwoTower, build_rows, load_data  # noqa: E402
from run_v2 import ListwiseReranker, build_features, hr_ndcg  # noqa: E402

OUT_DIR = os.environ.get("LEGR_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------- 召回
@torch.no_grad()
def recall_topk(model, emb_t, inter, uids, k=50, max_hist=20, bs=256):
    """返回 {uid: (pool[np], prior[np], tpos)}，prior 是双塔分数"""
    D = emb_t.size(1)
    hist = np.zeros((len(uids), max_hist), dtype=np.int64)
    mask = np.zeros((len(uids), max_hist), dtype=bool)
    tgt = np.zeros(len(uids), dtype=np.int64)
    for i, u in enumerate(uids):
        h = inter[u][:-1][-max_hist:]
        hist[i, max_hist - len(h):] = h
        mask[i, max_hist - len(h):] = True
        tgt[i] = inter[u][-1]
    hist_t, mask_t, tgt_t = torch.from_numpy(hist), torch.from_numpy(mask), torch.from_numpy(tgt)

    I = model.item(emb_t)                                  # (N, d)
    out = {}
    for s in range(0, len(uids), bs):
        h, m, t = hist_t[s:s + bs], mask_t[s:s + bs], tgt_t[s:s + bs]
        u_vec = model.user(emb_t[h] * m.unsqueeze(-1), m)
        score = u_vec @ I.t()                              # (B, N)
        for i in range(h.size(0)):
            score[i, h[i][m[i]]] = -1e9                    # 屏蔽历史
            top = score.topk(k, dim=1).indices[i]
            pool = top.numpy()
            prior = score[i, top].cpu().numpy()
            hit = (pool == t[i].item())
            tpos = int(np.where(pool == t[i].item())[0][0]) if hit.any() else -1
            out[uids[s + i]] = {"pool": pool, "prior": prior, "tpos": tpos, "hit": bool(hit.any())}
    return out


# ---------------------------------------------------------------- 数据
def make_pool_rows(users, rec, inter, train=True, k=50):
    """训练：pool = [target] + 召回 Top-50；测试：pool = 纯 Top-50"""
    rows = []
    for u in users:
        d = rec[u]
        if train:
            target = int(inter[u][-1])
            pool = np.asarray(d["pool"], dtype=np.int64)[:k]
            prior = np.asarray(d["prior"], dtype=np.float32)[:k]
            if target in pool:
                # target 本来就被召回：直接用真实候选池，tpos 指向它
                tpos = int(np.where(pool == target)[0][0])
            else:
                # 没被召回：把 target 放进池子（长度仍为 k），训练需要正例
                pool = np.concatenate([[target], pool[:k - 1]])
                prior = np.concatenate([[prior[:k - 1].max()], prior[:k - 1]]).astype(np.float32)
                tpos = 0
            rows.append({"uid": u, "pool": pool, "prior": prior, "tpos": tpos})
        else:
            rows.append({"uid": u, "pool": d["pool"], "prior": d["prior"].astype(np.float32),
                         "tpos": d["tpos"]})
    return rows


class PoolDS(torch.utils.data.Dataset):
    def __init__(self, rows, uid2row, shuffle=False, seed=0):
        self.rows, self.uid2row, self.shuffle = rows, uid2row, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = self.rows[i]
        pool, prior, tpos = d["pool"], d["prior"], d["tpos"]
        if self.shuffle:
            p = self.rng.permutation(len(pool))
            pool, prior = pool[p], prior[p]
            tpos = int(np.where(p == tpos)[0][0])
        z = (prior - prior.mean()) / (prior.std() + 1e-6)
        return (torch.from_numpy(pool.astype(np.int64)),
                torch.from_numpy(z.astype(np.float32)),
                self.uid2row[d["uid"]], int(tpos))


def train_eval(rows_tr, rows_va, rows_te, item_feat, user_feat, uid2row, args, seed=0):
    torch.manual_seed(seed)
    model = ListwiseReranker(item_feat.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    tr = PoolDS(rows_tr, uid2row, shuffle=True, seed=seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.bs, shuffle=True)

    def score_rows(rows):
        model.eval()
        ds = PoolDS(rows, uid2row)
        out = []
        with torch.no_grad():
            for pool, pz, urow, _ in torch.utils.data.DataLoader(ds, batch_size=512):
                s = model(torch.from_numpy(item_feat[pool.numpy()]), pz,
                          torch.from_numpy(user_feat[urow.numpy()]))
                out.append(s)
        return torch.cat(out).numpy()

    best = (-1, None)
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
        s = score_rows(rows_va)
        r = np.array([row["tpos"] for row in rows_va])
        hit = np.array([(r[i] >= 0) and (np.argsort(-s[i]).tolist().index(r[i]) < 10)
                        for i in range(len(rows_va))], dtype=np.float32)
        print(f"    ep{ep+1} loss={tot/len(dl):.4f} valid 条件HR@10={hit.mean():.4f}")
        if hit.mean() > best[0]:
            best = (hit.mean(), {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, score_rows(rows_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()

    inter, emb, seq = load_data()
    emb_t = torch.from_numpy(emb)
    item_feat, user_feat, uids = build_features(inter, emb, emb.shape[0])
    uid2row = {u: i for i, u in enumerate(uids.tolist())}

    # 1) 双塔召回
    ck = torch.load(os.path.join(OUT_DIR, "two_tower.pt"), map_location="cpu")
    a = ck["args"]
    tt = TwoTower(emb.shape[1], a["d"], a["pool"])
    tt.load_state_dict(ck["state_dict"])
    tt.eval()

    all_users = [r["uid"] for s in ("train", "valid", "test") for r in seq[s]]
    print(f"召回 {len(all_users)} 个用户 Top-{args.k} ...")
    rec = recall_topk(tt, emb_t, inter, all_users, k=args.k)
    tr_u = [r["uid"] for r in seq["train"]]
    va_u = [r["uid"] for r in seq["valid"]]
    te_u = [r["uid"] for r in seq["test"]]
    print(f"召回完成，耗时 {time.time()-t0:.0f}s")

    rows_tr = make_pool_rows(tr_u, rec, inter, train=True, k=args.k)
    rows_va = make_pool_rows(va_u, rec, inter, train=False)
    rows_te = make_pool_rows(te_u, rec, inter, train=False)

    # 2) 基线：不重排
    recall_rate = np.mean([r["tpos"] >= 0 for r in rows_te])
    prior_hit, prior_top10 = [], []
    for r in rows_te:
        if r["tpos"] < 0:
            prior_hit.append(0); continue
        rank = int(np.argsort(-r["prior"]).tolist().index(r["tpos"])) + 1
        prior_hit.append(float(rank <= 10))
        prior_top10.append(rank)
    prior_e2e = float(np.mean(prior_hit))

    # 3) 训练重排并评估
    print("\n训练重排器 ...")
    hits = []
    for sd in args.seeds:
        model, s = train_eval(rows_tr, rows_va, rows_te, item_feat, user_feat, uid2row, args, sd)
        hit = []
        for i, r in enumerate(rows_te):
            if r["tpos"] < 0:
                hit.append(0.0); continue
            rank = int(np.argsort(-s[i]).tolist().index(r["tpos"])) + 1
            hit.append(float(rank <= 10))
        hits.append(np.array(hit))
        print(f"  seed{sd}: 端到端 HR@10={np.mean(hit):.4f}")

    h = np.mean(hits, axis=0)
    cond = np.array([r["tpos"] >= 0 for r in rows_te])

    def ci(x, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print("\n" + "=" * 70)
    print("端到端（全库 24,587 → 召回 50 → 重排 → Top-10）")
    print("=" * 70)
    print(f"召回率 Recall@50        = {recall_rate:.4f}")
    print(f"不重排 端到端 HR@10      = {prior_e2e:.4f}")
    print(f"重排后 端到端 HR@10      = {h.mean():.4f}  95%CI {ci(h)}")
    print(f"条件 HR@10（仅召回命中）  = {h[cond].mean():.4f}  (n={int(cond.sum())})")
    print("\n对照：SASRec 单模型全库 HR@10 = 0.0314")
    print(f"     语义召回+重排（旧）      ≈ 0.026")

    with open(os.path.join(OUT_DIR, "end_to_end_report.json"), "w", encoding="utf-8") as f:
        json.dump({"recall50": float(recall_rate), "e2e_hr10": float(h.mean()),
                   "e2e_hr10_ci": ci(h), "e2e_hr10_no_rerank": prior_e2e,
                   "cond_hr10": float(h[cond].mean()), "n_test": len(rows_te),
                   "args": vars(args)}, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
