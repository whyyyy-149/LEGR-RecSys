import json
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ITEM2IDX_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item2idx.json"
META_PATH = r"C:\Users\wuhan\LEGR\data\raw\meta\meta_Musical_Instruments_clean.jsonl"
OUT_PATH = r"C:\Users\wuhan\LEGR\data\processed\Musical_Instruments\item_embeddings.npy"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16 

# ---------- 1. 读 item2idx ----------
with open(ITEM2IDX_PATH, "r") as f:
    item2idx = json.load(f)
print(f"item2idx 物品数: {len(item2idx)}")

# ---------- 2. 读 metadata，构建 asin -> text ----------
asin2text = {}
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        asin = obj.get("parent_asin")
        if not asin:
            continue
        title = (obj.get("title") or "").strip()
        desc_list = obj.get("description") or []
        if isinstance(desc_list, str):
            desc_list = [desc_list]
        desc = " ".join(x.strip() for x in desc_list if isinstance(x, str) and x.strip())
        if desc:
            text = f"{title}. {desc}".strip(". ")
        else:
            text = title
        asin2text[asin] = text

print(f"metadata 已解析: {len(asin2text)} 条")

# ---------- 3. 按 item2idx 顺序构造文本列表 ----------
ordered_asins = list(item2idx.keys())
texts = []
empty_text_count = 0
for asin in ordered_asins:
    t = asin2text.get(asin, "")
    if not t:
        empty_text_count += 1
    texts.append(t)

print(f"待编码文本数: {len(texts)}")
print(f"空文本数: {empty_text_count}（将用零向量填充）")

lengths = [len(t) for t in texts]
print(f"文本字符长度: 平均 {int(np.mean(lengths))}, 最大 {max(lengths)}")

# ---------- 4. 加载模型 ----------
print(f"加载模型: {MODEL_NAME}")
t0 = time.time()
# 移除 max_seq_length 参数，只保留 float32
model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float32}
)
# 如果版本支持，再用 setter 设置；如果不支持，这一步会跳过，靠后面的 Python 硬截断兜底
if hasattr(model, "max_seq_length"):
    model.max_seq_length = 512
print(f"模型加载耗时: {time.time() - t0:.1f}s")

# ---------- 5. 编码 ----------
# 【核心提速】在 Python 层强制把超长文本截断到前 1000 个字符
# 这样模型就不会去处理那个 68265 字的极端长文本了
encode_texts = [t[:1000] if t else " " for t in texts]

t0 = time.time()
embeddings = model.encode(
    encode_texts,
    batch_size=BATCH_SIZE, # 保持 16
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
elapsed = time.time() - t0
print(f"编码耗时: {elapsed:.1f}s，平均 {elapsed / len(texts) * 1000:.1f} ms/条")

# ---------- 6. 空文本位置置零 ----------
zero_rows = [i for i, t in enumerate(texts) if not t]
if zero_rows:
    embeddings[zero_rows] = 0.0
    print(f"已将 {len(zero_rows)} 行空文本置为零向量")

# ---------- 7. 保存 ----------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, embeddings)
size_mb = os.path.getsize(OUT_PATH) / 1024**2
print(f"保存到: {OUT_PATH}")
print(f"shape: {embeddings.shape}")
print(f"文件大小: {size_mb:.2f} MB")
print(f"dtype: {embeddings.dtype}")