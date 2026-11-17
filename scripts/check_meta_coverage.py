import json

META_PATH = r"C:\Users\wuhan\LEGR\data\raw\meta\meta_Musical_Instruments_clean.jsonl"
ITEM2IDX_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item2idx.json"

# ---------- 第一部分：metadata 自身的覆盖率 ----------
count = has_title = has_desc = 0
meta_asins = set()

with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        count += 1
        if obj.get("title"):
            has_title += 1
        if obj.get("description"):
            has_desc += 1
        if obj.get("parent_asin"):
            meta_asins.add(obj["parent_asin"])

print("===== metadata 覆盖率体检 =====")
print(f"条数: {count}")
print(f"title 覆盖率: {has_title / count:.3f}")
print(f"description 覆盖率: {has_desc / count:.3f}")
print(f"metadata 中唯一 parent_asin 数: {len(meta_asins)}")

# ---------- 第二部分：metadata 对 item2idx 的覆盖率 ----------
with open(ITEM2IDX_PATH, "r") as f:
    item2idx = json.load(f)

item_asins = set(item2idx.keys())
covered = item_asins & meta_asins

print("\n===== item2idx 物品覆盖率 =====")
print(f"item2idx 物品数: {len(item_asins)}")
print(f"metadata 覆盖的物品数: {len(covered)}")
print(f"覆盖率: {len(covered) / len(item_asins):.3f}")
print(f"未覆盖的物品数: {len(item_asins - meta_asins)}")

if item_asins - meta_asins:
    sample_missing = list(item_asins - meta_asins)[:5]
    print(f"未覆盖样例（前 5 个）: {sample_missing}")