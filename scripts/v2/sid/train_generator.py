"""
Stage 2-B：训练生成式检索模型（TIGER-lite）

任务：给定用户历史的 SID token 序列（如 10 个物品 × 4 码 = 40 个 token），
      自回归生成下一个物品的 4 个码字。

训练方式（teacher forcing）：
    x = [hist(40), t0, t1, t2]        → 长度 43
    用位置 39..42 的输出分别预测 t0..t3
    只在这 4 个位置算交叉熵

用法：
  python scripts/v2/sid/train_generator.py --epochs 5
  python scripts/v2/sid/train_generator.py --epochs 2 --bs 128     # 冒烟
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
SEQ_PATH = os.path.join(HERE, "outputs", "sid_seq.pkl")
OUT_DIR = os.environ.get("LEGR_SID_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


class SIDGenerator(nn.Module):
    """一个标准的 causal Transformer + 词表头，没有花活。
    和 SASRec 的唯一区别：SASRec 输出一个打分，这里输出 257 类上的分布。"""

    def __init__(self, vocab=257, d=128, heads=4, layers=4, max_len=64, dropout=0.1, pad=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        L = x.size(1)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        # 因果掩码：第 i 个位置只能看到 0..i
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=causal)
        return self.out(self.norm(h))


def make_arrays(recs, L, n_levels):
    n = len(recs)
    hist = np.full((n, L), 256, dtype=np.int64)          # 左填充 PAD=256
    tgt = np.zeros((n, n_levels), dtype=np.int64)
    for i, r in enumerate(recs):
        h = r["hist"][-L:]
        hist[i, L - len(h):] = h
        tgt[i] = r["target_codes"]
    return torch.from_numpy(hist), torch.from_numpy(tgt)


@torch.no_grad()
def evaluate(model, hist, tgt, bs=512):
    model.eval()
    n_levels = tgt.size(1)
    correct = torch.zeros(n_levels, dtype=torch.long)
    exact = 0
    for s in range(0, hist.size(0), bs):
        h, t = hist[s:s + bs], tgt[s:s + bs]
        x = torch.cat([h, t[:, :-1]], dim=1)
        logits = model(x)[:, -n_levels:, :]
        pred = logits.argmax(-1)
        correct += (pred == t).sum(0)
        exact += (pred == t).all(1).sum().item()
    n = hist.size(0)
    return {
        "per_level_acc": (correct.float() / n).tolist(),
        "exact_match": exact / n,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    with open(SEQ_PATH, "rb") as f:
        data = pickle.load(f)
    meta = data["meta"]
    L, n_levels, vocab = meta["max_hist_tokens"], meta["n_levels"], meta["vocab"]
    print(f"max_hist_tokens={L}  n_levels={n_levels}  vocab={vocab}")

    tr_h, tr_t = make_arrays(data["train"], L, n_levels)
    va_h, va_t = make_arrays(data["valid"], L, n_levels)
    te_h, te_t = make_arrays(data["test"], L, n_levels)
    print(f"train/valid/test = {len(tr_h)}/{len(va_h)}/{len(te_h)}")

    model = SIDGenerator(vocab, args.d, args.heads, args.layers, L + 4, pad=meta["pad"])
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_h))
        tot = 0.0
        for s in range(0, len(tr_h), args.bs):
            idx = perm[s:s + args.bs]
            h, t = tr_h[idx], tr_t[idx]
            x = torch.cat([h, t[:, :-1]], dim=1)          # (B, L+3)
            logits = model(x)[:, -n_levels:, :]           # 只取要预测的 4 个位置
            loss = F.cross_entropy(logits.reshape(-1, vocab), t.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        vm = evaluate(model, va_h, va_t)
        print(f"ep{ep+1:2d} loss={tot/(len(tr_h)//args.bs+1):.4f} "
              f"valid exact={vm['exact_match']:.4f} "
              f"acc={'/'.join(f'{a:.3f}' for a in vm['per_level_acc'])}")
        if vm["exact_match"] > best[0]:
            best = (vm["exact_match"], {k: v.detach().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    tm = evaluate(model, te_h, te_t)
    print(f"\n测试集 teacher-forcing: exact={tm['exact_match']:.4f} "
          f"acc={'/'.join(f'{a:.3f}' for a in tm['per_level_acc'])}")
    print("注意：这是 teacher-forcing 下的逐码准确率，真实检索效果要看 decode.py 的 Recall@50")

    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "meta": meta, "valid_exact": best[0]},
               os.path.join(OUT_DIR, "generator.pt"))
    with open(os.path.join(OUT_DIR, "generator_report.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": meta, "valid": best[0],
                   "test_teacher_forcing": tm, "elapsed_s": round(time.time() - t0, 1)},
                  f, ensure_ascii=False, indent=2)
    print(f"已保存 -> {OUT_DIR}/generator.pt  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
