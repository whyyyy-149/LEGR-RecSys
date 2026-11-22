"""
Stage 2-C：trie 约束的 beam search + 检索评估

为什么必须加 trie 约束：
  4 级 × 256 码 = 43 亿种组合，但真实物品只有 2.4 万个。
  不加约束，模型会大量生成"不存在的物品"（无效生成率常见 30%~60%），
  加了之后，每一步只允许走真实 SID 的前缀，无效生成率恒为 0。

用法：
  python scripts/v2/sid/decode.py --n_users 500 --beam 10        # 冒烟
  python scripts/v2/sid/decode.py --beam 50                      # 全量测试集
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import sys
sys.path.insert(0, HERE)
from train_generator import SIDGenerator  # noqa: E402

SID_CODES = os.path.join(HERE, "outputs", "sid_codes.npy")
SEQ_PATH = os.path.join(HERE, "outputs", "sid_seq.pkl")
CKPT = os.path.join(HERE, "outputs", "generator.pt")
OUT_DIR = os.environ.get("LEGR_SID_OUT", os.path.join(HERE, "outputs"))
ITEMS = "_items"


def build_trie(codes):
    """前缀树：路径是 SID，叶子挂物品 idx 列表（处理 3.3% 的重名）"""
    trie = {}
    for item, code in enumerate(codes):
        node = trie
        for c in code:
            node = node.setdefault(int(c), {})
        node.setdefault(ITEMS, []).append(int(item))
    return trie


def node_of(trie, codes):
    node = trie
    for c in codes:
        node = node.get(c)
        if node is None:
            return None
    return node


@torch.no_grad()
def beam_search(model, hist, trie, beam, n_steps, bs_users=64):
    """返回每个用户的 [(item, score), ...]，已按分数降序、按 item 去重"""
    results = []
    U_all = hist.size(0)
    for s in range(0, U_all, bs_users):
        h = hist[s:s + bs_users]
        U = h.size(0)
        beams = [[((), 0.0)] for _ in range(U)]

        for step in range(n_steps):
            seqs, owners, cands = [], [], []
            for u in range(U):
                for codes, score in beams[u]:
                    seq = torch.cat([h[u], torch.tensor(codes, dtype=torch.long)])
                    seqs.append(seq)
                    owners.append(u)
                    cands.append((codes, score))
            x = torch.stack(seqs)                                  # (N, L+step)
            logits = model(x)[:, -1, :]                            # (N, V)
            logp = F.log_softmax(logits, dim=-1)[:, :256]          # 只看码字部分

            new_beams = [[] for _ in range(U)]
            for i, u in enumerate(owners):
                codes, score = cands[i]
                node = node_of(trie, codes)
                if node is None:
                    continue
                for c, child in node.items():
                    if c == ITEMS:
                        continue
                    new_beams[u].append((codes + (c,), score + logp[i, c].item()))

            beams = [sorted(b, key=lambda t: -t[1])[:beam] for b in new_beams]

        # 叶子 → 物品
        out = []
        for u in range(U):
            item_score = {}
            for codes, score in beams[u]:
                node = node_of(trie, codes)
                if node is None or ITEMS not in node:
                    continue
                for it in node[ITEMS]:
                    item_score[it] = max(item_score.get(it, -1e18), score)
            ranked = sorted(item_score.items(), key=lambda kv: -kv[1])
            out.append(ranked)
        results.extend(out)
    return results


@torch.no_grad()
def invalid_rate(model, hist, trie, n_steps, n_users=100, beam=1):
    """不加约束时的无效生成率（教学用：对比加 trie 前后）"""
    h = hist[:n_users]
    devices = torch.arange(h.size(0))
    cur = torch.zeros(h.size(0), 0, dtype=torch.long)
    invalid = 0
    for _ in range(n_steps):
        x = torch.cat([h, cur], dim=1)
        nxt = model(x)[:, -1, :256].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    for codes in cur.tolist():
        node = node_of(trie, codes)
        if node is None or ITEMS not in node:
            invalid += 1
    return invalid / h.size(0)


def popularity_baseline(test, k=50):
    """热度基线：把全站最热的前 K 个物品（排除该用户历史）推给所有人。
    这是任何召回模型必须跨过的门槛——Amazon 5-core 上它往往高得吓人。"""
    import json
    inter_path = os.path.join(
        os.environ.get("LEGR_DATA",
                       os.path.join(REPO_ROOT, "data", "processed", "Musical_Instruments")),
        "musical_instruments.inter.json")
    with open(inter_path, "r", encoding="utf-8") as f:
        inter = json.load(f)
    n_items = int(np.load(SID_CODES, mmap_mode="r").shape[0])
    cnt = np.zeros(n_items)
    for s in inter.values():
        for i in s:
            cnt[i] += 1
    order = np.argsort(-cnt)
    hit50 = hit10 = 0
    for r in test:
        hist = set(inter[r["uid"]][:-1])
        rec = [it for it in order if it not in hist][:k]
        t = r["target"]
        hit50 += t in rec
        hit10 += t in rec[:10]
    n = len(test)
    return hit50 / n, hit10 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--n_users", type=int, default=0, help="只评估前 N 个测试用户")
    ap.add_argument("--bs_users", type=int, default=64)
    ap.add_argument("--invalid_sample", type=int, default=100)
    ap.add_argument("--shuffle_targets", action="store_true",
                    help="负对照：打乱 target，指标应掉到 ~0；否则说明有泄漏/对齐 bug")
    args = ap.parse_args()

    t0 = time.time()
    codes = np.load(SID_CODES)
    trie = build_trie(codes)
    with open(SEQ_PATH, "rb") as f:
        data = pickle.load(f)
    meta = data["meta"]
    L, n_levels = meta["max_hist_tokens"], meta["n_levels"]

    ck = torch.load(CKPT, map_location="cpu")
    a = ck["args"]
    model = SIDGenerator(meta["vocab"], a["d"], a["heads"], a["layers"], L + 4, pad=meta["pad"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    test = data["test"]
    if args.n_users:
        test = test[:args.n_users]
    hist = np.full((len(test), L), meta["pad"], dtype=np.int64)
    for i, r in enumerate(test):
        h = r["hist"][-L:]
        hist[i, L - len(h):] = h
    hist = torch.from_numpy(hist)
    targets = np.array([r["target"] for r in test])
    if args.shuffle_targets:
        rng = np.random.default_rng(0)
        targets = targets[rng.permutation(len(targets))]
        print("[负对照] target 已打乱，指标应掉到 ~0")

    print(f"测试用户 {len(test)}，beam={args.beam}，SID 级数={n_levels}")
    ranked = beam_search(model, hist, trie, args.beam, n_levels, args.bs_users)

    hit50 = np.zeros(len(test), dtype=np.float32)
    hit10 = np.zeros(len(test), dtype=np.float32)
    n_cand = []
    for i, cand in enumerate(ranked):
        items = [it for it, _ in cand]
        n_cand.append(len(items))
        hit50[i] = float(targets[i] in items[:50])
        hit10[i] = float(targets[i] in items[:10])

    inv = invalid_rate(model, hist, trie, n_levels, args.invalid_sample) if args.invalid_sample else float("nan")
    pop50, pop10 = popularity_baseline(test)

    def ci(x, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print("\n=== 生成式检索结果 ===")
    print(f"Recall@50 = {hit50.mean():.4f}  95%CI [{ci(hit50)[0]:.4f}, {ci(hit50)[1]:.4f}]")
    print(f"HR@10     = {hit10.mean():.4f}  95%CI [{ci(hit10)[0]:.4f}, {ci(hit10)[1]:.4f}]")
    print(f"平均候选数 = {np.mean(n_cand):.1f}")
    print(f"无效生成率（不加 trie）= {inv:.1%}" if args.invalid_sample else "")
    print("\n对照（同一批用户口径，全库 24,587 候选）：")
    print(f"  热度基线（Top-50 热门，排除历史） Recall@50 = {pop50:.4f}  HR@10 = {pop10:.4f}")
    print("  纯语义 FAISS 召回                Recall@50 = 0.0432")
    print("  纯协同 SASRec 全量               Recall@50 = 0.0848  HR@10 = 0.0314")

    with open(os.path.join(OUT_DIR, "decode_report.json"), "w", encoding="utf-8") as f:
        json.dump({"n_users": len(test), "beam": args.beam,
                   "recall50": float(hit50.mean()), "recall50_ci": ci(hit50),
                   "hr10": float(hit10.mean()), "hr10_ci": ci(hit10),
                   "avg_candidates": float(np.mean(n_cand)),
                   "invalid_rate_unconstrained": None if inv != inv else float(inv),
                   "elapsed_s": round(time.time() - t0, 1)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果 -> {OUT_DIR}/decode_report.json")


if __name__ == "__main__":
    main()
