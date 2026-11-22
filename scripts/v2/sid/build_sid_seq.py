"""
Stage 2-A：把用户历史转成 SID token 序列

输入：inter.json（用户序列）+ sid_codes.npy（每个物品的 SID）
输出：outputs/sid_seq.pkl
      {"train": [...], "valid": [...], "test": [...], "meta": {...}}

每个样本：
    hist          历史物品的 SID 展平，如 [c1,c2,c3,c4, c1,c2,...]，最多 4*max_hist 个
    target        目标物品的 item idx
    target_codes  目标物品的 4 个码字（生成目标）

用法：
  python scripts/v2/sid/build_sid_seq.py
  python scripts/v2/sid/build_sid_seq.py --max_hist 8 --limit 3000   # 冒烟
"""
import argparse
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get(
    "LEGR_DATA", os.path.join(REPO_ROOT, "data", "processed", "Musical_Instruments")
)
INTER_PATH = os.path.join(DATA_ROOT, "musical_instruments.inter.json")
SID_CODES = os.path.join(HERE, "outputs", "sid_codes.npy")
OUT_PATH = os.path.join(HERE, "outputs", "sid_seq.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_hist", type=int, default=10, help="历史最多保留几个物品")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个用户（冒烟）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    codes = np.load(SID_CODES).astype(np.int64)
    n_levels = codes.shape[1]
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        inter = json.load(f)

    uids = list(inter.keys())
    if args.limit:
        uids = uids[:args.limit]

    recs = []
    for uid in uids:
        seq = inter[uid]
        if len(seq) < 2:
            continue
        hist, target = seq[:-1], seq[-1]
        hist = hist[-args.max_hist:]
        recs.append({
            "uid": uid,
            "hist": codes[hist].reshape(-1).tolist(),          # (<= 4*max_hist,)
            "target": int(target),
            "target_codes": codes[target].tolist(),            # (n_levels,)
        })

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    a, b = int(0.8 * len(perm)), int(0.9 * len(perm))
    split = {
        "train": [recs[i] for i in perm[:a]],
        "valid": [recs[i] for i in perm[a:b]],
        "test": [recs[i] for i in perm[b:]],
    }
    meta = {
        "max_hist": args.max_hist,
        "max_hist_tokens": args.max_hist * n_levels,
        "n_levels": n_levels,
        "n_items": int(codes.shape[0]),
        "vocab": 256 + 1,          # 0..255 码字 + PAD=256
        "pad": 256,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump({**split, "meta": meta}, f)

    hist_len = [len(r["hist"]) for r in recs]
    print(f"样本 {len(recs)}  train/valid/test = "
          f"{len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
    print(f"历史 token 长度: 平均 {np.mean(hist_len):.1f}  最大 {max(hist_len)}")
    print(f"SID 级数 {n_levels}，词表 {meta['vocab']}（PAD={meta['pad']}）")
    print(f"已保存 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
