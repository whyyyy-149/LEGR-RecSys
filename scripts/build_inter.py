import pandas as pd
import json
from pathlib import Path

base = r"C:\Users\wuhan\LEGR\data\raw\benchmark\5core"
out_dir = Path(r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments")
out_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(f"{base}/Musical_Instruments.train.csv.gz", encoding="utf-8-sig")
valid = pd.read_csv(f"{base}/Musical_Instruments.valid.csv.gz", encoding="utf-8-sig")
test  = pd.read_csv(f"{base}/Musical_Instruments.test.csv.gz",  encoding="utf-8-sig")

def parse_history(h):
    if pd.isna(h):
        return []
    if isinstance(h, str):
        h = h.strip()
        if not h or h.lower() == "nan":
            return []
        return h.split()
    return list(h)

full = pd.concat([train, valid, test], ignore_index=True)
full = full.dropna(subset=["parent_asin"]).copy()
full["history_list"] = full["history"].apply(parse_history)
full = full.sort_values(["user_id", "timestamp"])

# item2idx：覆盖 history 里所有物品 + 所有 parent_asin
all_items = set()
for seq in full["history_list"]:
    all_items.update(seq)
all_items.update(full["parent_asin"].unique())
all_items = {x for x in all_items if isinstance(x, str)}
item2idx = {item: idx for idx, item in enumerate(sorted(all_items))}

# 每个用户取 timestamp 最大的那行，history + parent_asin，去重保序
inter = {}
for user_id, group in full.groupby("user_id"):
    group = group.sort_values("timestamp")
    last = group.iloc[-1]
    seq = list(last["history_list"]) + [last["parent_asin"]]
    seen = set()
    seq_dedup = []
    for x in seq:
        if isinstance(x, str) and x not in seen:
            seen.add(x)
            seq_dedup.append(x)
    seq_dedup = [x for x in seq_dedup if x in item2idx]
    if len(seq_dedup) < 2:
        continue
    inter[str(user_id)] = [item2idx[x] for x in seq_dedup]

with open(out_dir / "musical_instruments.inter.json", "w") as f:
    json.dump(inter, f)
with open(out_dir / "item2idx.json", "w") as f:
    json.dump(item2idx, f)

print(f"用户数: {len(inter)}")
print(f"物品数: {len(item2idx)}")
if inter:
    print(f"平均序列长度: {sum(len(v) for v in inter.values()) / len(inter):.2f}")
    print(f"最短序列: {min(len(v) for v in inter.values())}")
    print(f"最长序列: {max(len(v) for v in inter.values())}")
    print(f"序列<2的用户数: {sum(1 for v in inter.values() if len(v) < 2)}")
    print("样例:", list(inter.items())[:2])