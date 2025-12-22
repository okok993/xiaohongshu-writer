import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv('聚类数据.csv')
print("数据集前5行：")
print(df.head())
print("\n数据集信息：")
print(df.info())

# 2. 可视化原始数据（人眼分析）
plt.figure(figsize=(10, 6))
plt.scatter(df['振动频率(Hz)'], df['温度(℃)'], c='blue', alpha=0.7, edgecolors='k')
plt.title('原始数据散点图（振动频率 vs 温度）')
plt.xlabel('振动频率(Hz)')
plt.ylabel('温度(℃)')
plt.grid(True)
plt.show()

# 3. 标准化数据
features = ['振动频率(Hz)', '温度(℃)', '压力(kPa)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans 聚类，K=2,3,4，初始化方法为 random
K_values = [2, 3, 4]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, init='random', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    axes[idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    axes[idx].set_title(f'K = {K} (random init)')
    axes[idx].set_xlabel('标准化振动频率')
    axes[idx].set_ylabel('标准化温度')
plt.tight_layout()
plt.show()

# 5. 选择最佳 K（使用轮廓系数）
from sklearn.metrics import silhouette_score

best_k = 0
best_score = -1
for K in range(2, 11):
    kmeans = KMeans(n_clusters=K, init='random', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f'K = {K}, 轮廓系数 = {score:.4f}')
    if score > best_score:
        best_score = score
        best_k = K
print(f'\n最佳 K = {best_k}，轮廓系数 = {best_score:.4f}')

# 6. 在最佳 K 下，比较 random 和 k-means++ 初始化
best_K = best_k

# random init
kmeans_random = KMeans(n_clusters=best_K, init='random', random_state=42, n_init=10)
labels_random = kmeans_random.fit_predict(X_scaled)

# k-means++ init
kmeans_plus = KMeans(n_clusters=best_K, init='k-means++', random_state=42, n_init=10)
labels_plus = kmeans_plus.fit_predict(X_scaled)

# 可视化比较
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_random, cmap='viridis', alpha=0.7, edgecolors='k')
axes[0].set_title(f'K = {best_K}，初始化方法：random')
axes[0].set_xlabel('标准化振动频率')
axes[0].set_ylabel('标准化温度')

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_plus, cmap='viridis', alpha=0.7, edgecolors='k')
axes[1].set_title(f'K = {best_K}，初始化方法：k-means++')
axes[1].set_xlabel('标准化振动频率')
plt.tight_layout()
plt.show()

# 7. 输出质心位置比较
print("\n质心位置比较：")
print("Random 初始化质心：")
print(kmeans_random.cluster_centers_)
print("\nK-means++ 初始化质心：")
print(kmeans_plus.cluster_centers_)

# 8. 计算并打印聚类中心数量差异
print("\n两种初始化方法的质心位置差异（欧氏距离）：")
for i in range(best_K):
    dist = np.linalg.norm(kmeans_random.cluster_centers_[i] - kmeans_plus.cluster_centers_[i])
    print(f'簇 {i} 质心距离 = {dist:.4f}')