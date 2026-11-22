"""
实验 2：SASRec 全量 ranking（24k 全库），给出与论文可对照的口径。

全量排序口径：
  - 对每个测试用户，用 SASRec 给全部 24,587 个物品打分
  - 屏蔽该用户历史中出现过的物品（标准做法）
  - 取 target 的排名，算 HR@10 / NDCG@10 / Recall@50

同时给出纯语义召回的 Recall@50（4.32%）作为对照。

用法：python scripts/v2/analyze_full_rank.py
输出：scripts/v2/outputs/full_rank.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pool import SASRec  # noqa: E402
from run_v2 import DATA_ROOT, OUT_DIR, REPO_ROOT, _load_pickle  # noqa: E402

INTER_PATH = os.path.join(DATA_ROOT, "musical_instruments.inter.json")
EMB_PATH = os.path.join(DATA_ROOT, "item_embeddings.npy")
CKPT = os.environ.get(
    "LEGR_SASREC", os.path.join(REPO_ROOT, "outputs", "SASRec", "best_Musical_Instruments.pt")
)
BATCH = 256


def main():
    t0 = time.time()
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        inter = json.load(f)
    n_items = np.load(EMB_PATH, mmap_mode="r").shape[0]
    uids = list(inter.keys())
    print(f"物品 {n_items}，用户 {len(uids)}")

    model = SASRec()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    emb = model.item_emb.weight.detach().numpy()      # (24588, 64)，0 是 padding
    emb[0] = 0.0

    hits10 = np.zeros(len(uids), dtype=np.float32)
    ndcg10 = np.zeros(len(uids), dtype=np.float32)
    hits50 = np.zeros(len(uids), dtype=np.float32)
    ranks = np.zeros(len(uids), dtype=np.int32)

    with torch.no_grad():
        for s in range(0, len(uids), BATCH):
            batch = uids[s:s + BATCH]
            seqs, targets, hists = [], [], []
            for u in batch:
                seq = inter[u]
                seqs.append([x + 1 for x in seq[:-1][-50:]])
                targets.append(seq[-1])
                hists.append(set(seq[:-1]))
            L = max(len(x) for x in seqs)
            inp = torch.LongTensor([x + [0] * (L - len(x)) for x in seqs])
            urep = model(inp).numpy()                  # (B, 64)

            scores = urep @ emb.T                      # (B, 24588)，列 j = embedding 行 j
            for i, u in enumerate(batch):
                scores[i, [h + 1 for h in hists[i]]] = -1e9   # 屏蔽历史（注意 +1 偏移）
            scores[:, 0] = -1e9                              # 屏蔽 padding
            order = np.argsort(-scores, axis=1)
            tg = np.array(targets) + 1                       # target 的 embedding 行号
            pos = (order == tg[:, None]).argmax(1) + 1
            ranks[s:s + len(batch)] = pos

    hits10 = (ranks <= 10).astype(np.float32)
    hits50 = (ranks <= 50).astype(np.float32)
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0).astype(np.float32)

    def ci(x, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    r = {
        "n_users": int(len(uids)),
        "hr10": float(hits10.mean()),
        "hr10_ci": ci(hits10),
        "ndcg10": float(ndcg10.mean()),
        "recall50": float(hits50.mean()),
        "recall50_ci": ci(hits50),
        "median_rank": float(np.median(ranks)),
    }
    print("\n=== SASRec 全量 ranking（24,587 候选，屏蔽历史）===")
    print(f"HR@10      = {r['hr10']:.4f}  CI {r['hr10_ci']}")
    print(f"NDCG@10    = {r['ndcg10']:.4f}")
    print(f"Recall@50  = {r['recall50']:.4f}  CI {r['recall50_ci']}   "
          f"（纯语义召回对照：0.0432）")
    print(f"target 排名中位数 = {r['median_rank']:.0f}")

    with open(os.path.join(OUT_DIR, "full_rank.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
