import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_excel('data.xlsx', header=None)
print("数据形状:", df.shape)
print("数据前5行:")
print(df.head())
print("\n数据统计描述:")
print(df.describe())

# 2. 数据可视化 - 原始数据
plt.figure(figsize=(15, 10))
for i in range(df.shape[1]):
    plt.subplot(4, 3, i+1)
    plt.plot(df[i], linewidth=1)
    plt.title(f'特征 {i+1}')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('原始数据特征图.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Z-score归一化
scaler = StandardScaler()
X_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(X_normalized, columns=[f'Feature_{i+1}' for i in range(df.shape[1])])

print("\n归一化后数据统计:")
print(df_normalized.describe())

# 4. 创建目标变量
time_steps = len(df)
target = np.zeros(time_steps)
# 1-72时间是健康状态设置为1
target[:72] = 1.0
# 73-192时间是健康状态从1递减到0
for i in range(72, time_steps):
    target[i] = 1.0 - (i - 72) / (time_steps - 72)

# 可视化目标变量
plt.figure(figsize=(10, 4))
plt.plot(target, linewidth=2, color='red')
plt.title('健康状态目标变量 (1=健康, 0=故障)')
plt.xlabel('时间步')
plt.ylabel('健康状态')
plt.grid(True, alpha=0.3)
plt.axvline(x=72, color='green', linestyle='--', alpha=0.7, label='健康到退化转折点')
plt.legend()
plt.savefig('健康状态目标变量.png', dpi=300, bbox_inches='tight')
plt.show()