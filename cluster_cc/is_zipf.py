import faiss
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── 假设已有 kmeans 和 cluster_ids ──
from common import load_embedding

embeddings, parent_idx_l = load_embedding("/root/autodl-fs/bge-processed/_dense_sparse_court/")
embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
faiss.normalize_L2(embeddings)
print("embedding loaded.")

kmeans = faiss.Kmeans(d=1024, k=500, gpu=True, niter=20)
kmeans.train(embeddings)
# kmeans 训练完之后
_, cluster_ids = kmeans.index.search(embeddings, 1)
cluster_ids = cluster_ids.flatten()  # shape: (N,)
print("train doned.")


# 统计每个簇的大小
cluster_sizes = np.bincount(cluster_ids, minlength=500)
cluster_sizes_sorted = np.sort(cluster_sizes)[::-1]  # 从大到小排列

ranks = np.arange(1, len(cluster_sizes_sorted) + 1)

# ─────────────────────────────────────────
# 图1：簇大小分布（排名 vs 大小）
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 普通坐标
axes[0].bar(ranks, cluster_sizes_sorted, color="steelblue")
axes[0].set_xlabel("簇排名（从大到小）")
axes[0].set_ylabel("文档数量")
axes[0].set_title("簇大小分布")

# log-log 坐标（Zipf检验：如果是直线则符合幂律）
axes[1].loglog(ranks, cluster_sizes_sorted, "o", markersize=3, color="steelblue")
axes[1].set_xlabel("log(排名)")
axes[1].set_ylabel("log(文档数量)")
axes[1].set_title("Log-Log 图（直线=符合Zipf）")

# 拟合幂律直线
log_ranks = np.log(ranks)
log_sizes = np.log(cluster_sizes_sorted + 1)
coeffs = np.polyfit(log_ranks, log_sizes, 1)
axes[1].loglog(ranks, np.exp(np.polyval(coeffs, log_ranks)),
               "r--", label=f"幂律拟合 α={-coeffs[0]:.2f}")
axes[1].legend()

# 直方图
axes[2].hist(cluster_sizes, bins=50, color="steelblue", edgecolor="white")
axes[2].set_xlabel("簇内文档数")
axes[2].set_ylabel("簇的数量")
axes[2].set_title("簇大小直方图")

plt.tight_layout()
plt.savefig("cluster_zipf.png", dpi=150)
plt.show()

# ─────────────────────────────────────────
# 打印统计数据
# ─────────────────────────────────────────
print(f"总文档数:   {cluster_ids.shape[0]:,}")
print(f"簇数量:     {500}")
print(f"平均簇大小: {cluster_sizes.mean():.0f}")
print(f"中位数:     {np.median(cluster_sizes):.0f}")
print(f"最大簇:     {cluster_sizes.max():,}")
print(f"最小簇:     {cluster_sizes.min():,}")
print(f"标准差:     {cluster_sizes.std():.0f}")
print(f"幂律指数 α: {-coeffs[0]:.3f}  （接近1.0则符合Zipf）")

# 头部集中程度
top10 = cluster_sizes_sorted[:10].sum()
print(f"\n最大10个簇占总文档: {top10/cluster_ids.shape[0]*100:.1f}%")
top50 = cluster_sizes_sorted[:50].sum()
print(f"最大50个簇占总文档: {top50/cluster_ids.shape[0]*100:.1f}%")