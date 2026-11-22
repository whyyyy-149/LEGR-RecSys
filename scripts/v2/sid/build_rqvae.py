"""
Stage 1：RQ-VAE —— 把 1024 维文本向量压成 3 级语义 ID（SID）

产物：
  scripts/v2/sid/outputs/rqvae.pt        模型权重（含中心化用到的 mean）
  scripts/v2/sid/outputs/sid_codes.npy   (24587, 3) int16，每个物品的 (c1,c2,c3)
  scripts/v2/sid/outputs/sid_report.json 四个评估指标

用法：
  python scripts/v2/sid/build_rqvae.py --epochs 5 --subset 2000   # 冒烟
  python scripts/v2/sid/build_rqvae.py --epochs 200               # 全量

三个指标在文件末尾有详细说明。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get(
    "LEGR_DATA", os.path.join(REPO_ROOT, "data", "processed", "Musical_Instruments")
)
EMB_PATH = os.path.join(DATA_ROOT, "item_embeddings.npy")
OUT_DIR = os.environ.get("LEGR_SID_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------- 残差量化
class ResidualQuantizer(nn.Module):
    """残差量化（RQ）：

        r0 = z
        c1 = argmin_k ||r0 - e1_k|| ;  r1 = r0 - e1[c1]
        c2 = argmin_k ||r1 - e2_k|| ;  r2 = r1 - e2[c2]
        c3 = argmin_k ||r2 - e3_k||
        z_q = e1[c1] + e2[c2] + e3[c3]

    码本用梯度更新：
        codebook_loss = ||sg(r) - e||²   把码字拉向数据
        commitment    = ||r - sg(e)||²   把编码器输出拉向码字（β 加权）
    直通估计器：前向用 e，反向把梯度原样传给 r。
    """

    def __init__(self, n_levels=3, codebook_size=256, dim=64, beta=0.25):
        super().__init__()
        self.L, self.K, self.D, self.beta = n_levels, codebook_size, dim, beta
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim) * 0.1)
            for _ in range(n_levels)
        ])

    def forward(self, z, update=True):
        B = z.size(0)
        codes = z.new_zeros((B, self.L), dtype=torch.long)
        residual = z
        z_q = torch.zeros_like(z)
        loss_cb = z.new_zeros(())

        for l in range(self.L):
            e = self.codebooks[l]                                  # (K, D)
            # 距离展开，避免 (B, K, D) 大张量：||r||² - 2r·e + ||e||²
            d = (residual ** 2).sum(1, keepdim=True) \
                - 2 * residual @ e.t() + (e ** 2).sum(1)
            idx = d.argmin(1)
            codes[:, l] = idx
            q = e[idx]

            loss_cb = loss_cb + F.mse_loss(residual.detach(), q)   # 拉码字
            if update:
                loss_cb = loss_cb + self.beta * F.mse_loss(residual, q.detach())  # 拉编码器

            q_st = residual + (q - residual).detach()              # 直通
            z_q = z_q + q_st
            residual = residual - q                                # 残差继续量化

        return z_q, codes, loss_cb

    @torch.no_grad()
    def revive_dead(self, z_pool, thresh=0.5, max_revive=8):
        """死码重启：用随机样本重置这一轮几乎没用到的码字。
        阈值和数量都要保守，否则码本一直在抖、永远不收敛。"""
        revived = 0
        for l in range(self.L):
            e = self.codebooks[l]
            d = (z_pool ** 2).sum(1, keepdim=True) - 2 * z_pool @ e.t() + (e ** 2).sum(1)
            cnt = torch.bincount(d.argmin(1), minlength=self.K).float()
            dead = (cnt <= thresh).nonzero().flatten()
            for k in dead[:max_revive]:
                j = torch.randint(0, z_pool.size(0), (1,)).item()
                self.codebooks[l][k] = z_pool[j] + 0.01 * torch.randn_like(z_pool[j])
                revived += 1
        return revived


class RQVAE(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, latent=64, n_levels=3,
                 codebook_size=256, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, latent))
        self.rq = ResidualQuantizer(n_levels, codebook_size, latent, beta)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, in_dim))

    def forward(self, x, update=True):
        z = self.enc(x)
        z_q, codes, loss_cb = self.rq(z, update=update)
        return self.dec(z_q), codes, loss_cb, z


# ---------------------------------------------------------------- 指标
def codebook_stats(codes, K):
    """① 码本使用率：多少码字被用到、分布有多均匀"""
    out = []
    for l in range(codes.shape[1]):
        cnt = np.bincount(codes[:, l].astype(np.int64), minlength=K).astype(np.float64)
        p = cnt / cnt.sum()
        used = int((cnt > 0).sum())
        ppl = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        out.append({"level": l + 1, "used_codes": used, "usage_ratio": used / K,
                    "perplexity": ppl, "max_share": float(p.max())})
    return out


def collision_stats(codes):
    """② SID 冲突率：不同物品压成同一串码的比例"""
    n, L = codes.shape
    keys = codes[:, 0].astype(np.int64)
    for l in range(1, L):
        keys = keys * 256 + codes[:, l]
    uniq, cnt = np.unique(keys, return_counts=True)
    return {
        "n_items": int(n), "n_unique_sid": int(len(uniq)),
        "collision_rate": float(1 - len(uniq) / n),
        "items_in_collision": int((cnt[np.searchsorted(uniq, keys)] > 1).sum()),
        "largest_group": int(cnt.max()),
        "expected_random_collision": float(1 - np.exp(-n / (2 * 256 ** L))),
    }


@torch.no_grad()
def recon_stats(X, X_hat):
    """③ 重构余弦：压完再解压，能不能还原"""
    cos = F.cosine_similarity(X, X_hat, dim=1).numpy()
    return {"mean_cos": float(cos.mean()), "median_cos": float(np.median(cos)),
            "p10_cos": float(np.percentile(cos, 10))}


@torch.no_grad()
def prefix_neighbor_recall(X, codes, n_sample=500, k=10, seed=0):
    """④ 前缀邻域保持率：SID 的前缀有没有承载语义邻域
    对采样的物品取它在原空间的 top-k 近邻，统计它们与它在第 L 级前缀相同的比例。
    随机基线 = 1/256^L。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.size(0), size=min(n_sample, X.size(0)), replace=False)
    Xn = F.normalize(X, dim=1)
    sim = Xn[idx] @ Xn.t()
    sim[torch.arange(len(idx)), torch.from_numpy(idx)] = -2
    topk = sim.topk(k, dim=1).indices.numpy()

    res = {}
    for L in range(1, codes.shape[1] + 1):
        hit = 0.0
        for r, i in enumerate(idx):
            hit += (codes[topk[r], :L] == codes[i, :L]).all(1).mean()
        res[f"L{L}_neighbor_share"] = float(hit / len(idx))
        res[f"L{L}_random_baseline"] = float(256 ** (-L))
    return res


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=32,
                    help="latent 维度：越大重构越好但量化越粗（冲突率升高）")
    ap.add_argument("--n_levels", type=int, default=4)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    X0 = torch.from_numpy(np.load(EMB_PATH).astype(np.float32))
    if args.subset:
        X0 = X0[:args.subset]
    # 先减均值：这批向量和均值的余弦高达 0.67，公共分量会吃掉码本容量
    mu = X0.mean(0, keepdim=True)
    X = X0 - mu if not args.no_center else X0
    # 再按 RMS 归一，让各维尺度一致，训练更稳
    scale = X.std().item()
    X = X / scale
    X0n = X0 / X0.norm(dim=1, keepdim=True).clamp_min(1e-6)
    print(f"物品 {X.size(0)}，维度 {X.size(1)}，中心化={not args.no_center}，std={scale:.4f}")

    model = RQVAE(X.size(1), 512, args.latent, args.n_levels, args.codebook, args.beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = X.size(0)

    # 码本初始化：用当前 latent 的随机样本填充。
    # 不初始化的话，randn*0.1 的码字模长只有 0.8，而 latent 模长 ~8，
    # 量化损失会一直压着编码器收缩，重构学不动。
    with torch.no_grad():
        z_init = model.enc(X[torch.randperm(n)[:4096]])
        resid = z_init
        for l in range(args.n_levels):
            pick = torch.randint(0, resid.size(0), (args.codebook,))
            model.rq.codebooks[l].copy_(resid[pick] + 0.01 * torch.randn(args.codebook, args.latent))
            e = model.rq.codebooks[l]
            d = (resid ** 2).sum(1, keepdim=True) - 2 * resid @ e.t() + (e ** 2).sum(1)
            resid = resid - e[d.argmin(1)]
    print("码本已用 latent 样本初始化")

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot_r, tot_v = 0.0, 0.0
        z_pool = []
        for s in range(0, n, args.batch):
            xb = X[perm[s:s + args.batch]]
            x_hat, codes, loss_vq, z = model(xb)
            loss_r = F.mse_loss(x_hat, xb)
            loss = loss_r + loss_vq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_r += loss_r.item()
            tot_v += loss_vq.item()
            if len(z_pool) < 64:
                z_pool.append(z.detach())
        sched.step()
        revived = model.rq.revive_dead(torch.cat(z_pool)) if (z_pool and ep >= 10) else 0
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"ep{ep+1:3d} recon={tot_r/(n//args.batch+1):.5f} "
                  f"vq={tot_v/(n//args.batch+1):.4f} revived={revived}")

    # ---- 全量推理
    model.eval()
    codes_all, X_hat_all = [], []
    with torch.no_grad():
        for s in range(0, n, 1024):
            x_hat, codes, _, _ = model(X[s:s + 1024], update=False)
            codes_all.append(codes.numpy())
            X_hat_all.append(x_hat)
    codes = np.concatenate(codes_all).astype(np.int16)
    X_hat = torch.cat(X_hat_all)

    cb = codebook_stats(codes, args.codebook)
    co = collision_stats(codes)
    rc = recon_stats(X, X_hat)                       # 中心化+缩放空间（主指标）
    rc_orig = recon_stats(X0n, F.normalize(X_hat * scale + mu, dim=1))  # 原空间
    pn = prefix_neighbor_recall(X0n, codes, k=10)

    print("\n① 码本使用率（每级）")
    for r in cb:
        print(f"   L{r['level']}: used {r['used_codes']}/{args.codebook} "
              f"({r['usage_ratio']:.1%})  perplexity {r['perplexity']:.1f}  "
              f"max_share {r['max_share']:.1%}")

    print("\n② SID 冲突")
    print(f"   唯一 SID {co['n_unique_sid']}/{co['n_items']}，"
          f"冲突率 {co['collision_rate']:.4%}（理论随机值 {co['expected_random_collision']:.4%}），"
          f"最大同码组 {co['largest_group']}")

    print("\n③ 重构余弦")
    print(f"   中心化空间: mean {rc['mean_cos']:.4f}  median {rc['median_cos']:.4f}  "
          f"p10 {rc['p10_cos']:.4f}")
    print(f"   原空间    : mean {rc_orig['mean_cos']:.4f}")

    print("\n④ 前缀邻域保持（top-10 近邻中与它共享前缀的比例）")
    for L in range(1, args.n_levels + 1):
        print(f"   L{L}: {pn[f'L{L}_neighbor_share']:.2%}   "
              f"(随机基线 {pn[f'L{L}_random_baseline']:.2%})")

    np.save(os.path.join(OUT_DIR, "sid_codes.npy"), codes)
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "mu": mu, "scale": scale}, os.path.join(OUT_DIR, "rqvae.pt"))
    with open(os.path.join(OUT_DIR, "sid_report.json"), "w", encoding="utf-8") as f:
        json.dump({"n_items": int(n), "args": vars(args), "codebook": cb,
                   "collision": co, "recon_centered": rc, "recon_original": rc_orig,
                   "prefix_neighbor": pn, "elapsed_s": round(time.time() - t0, 1)},
                  f, ensure_ascii=False, indent=2)

    print(f"\n已保存：{OUT_DIR}/sid_codes.npy, rqvae.pt, sid_report.json")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()


# ================================================================
# 四个指标在评估什么（面试要能讲）
#
# ① 码本使用率 / 困惑度
#    问题：256 个码字，实际用了几个？分布有多均匀？
#    症状：只用 20 个码 → 「码本坍缩」，24k 物品挤在 20 个格子里，语义 ID 失去区分度。
#    好值：每级 used ≥ 90%、perplexity 接近 256、max_share < 5%。
#    本项目实测：4 级均 ≥86%，perplexity 200+，没有坍缩。
#
# ② SID 冲突率
#    问题：两个不同物品被压成同一串 (c1..cL) 的比例。
#    为什么致命：生成式检索里 SID 就是「物品的名字」，重名 = 检索不出来。
#    理论值：24,587 个物品落在 256^L 个码上，L=3 时碰撞期望 0.07%。
#    好值：<5%（TIGER 的做法是对冲突物品追加一个 token 消歧）。
#    实测：L=3 → 12%，L=4 → 3.3%。所以默认用 4 级。
#
# ③ 重构余弦
#    问题：编码再解码后，还像不像原来的向量；衡量这几个码字保住了多少信息。
#    注意：必须在中心化空间看。原空间里所有物品都有 0.67 的公共分量，
#    哪怕重构出「平均物品」也能拿到 0.86，那是假象。
#    好值：这里 0.72 属于「够用」。因为 1024 维向量的 top-100 PC 只解释 68% 方差，
#    32 维 latent 的信息上限本来就在 0.75 附近——想更高就得加大 latent，
#    但 latent 一大，同样 256 个码字在更高维空间里就更稀疏，冲突率立刻反弹。
#    这就是 RQ-VAE 最核心的权衡，面试问「为什么 3 级 / 为什么这个码本大小」就答这个。
#
# ④ 前缀邻域保持率（额外加的，比重构更贴近检索）
#    问题：语义上相近的两个物品，SID 前缀是否也相同？
#    随机基线 1/256 = 0.39%；L1 能到 74% 说明第一级码字真的在表达「粗类目」，
#    beam search 才能逐级收窄候选。
# ================================================================
