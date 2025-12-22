from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 线性回归降维
lr_model = LinearRegression()
lr_model.fit(X_normalized, target)
y_pred_lr = lr_model.predict(X_normalized)

# 2. 评估线性回归模型
mse_lr = mean_squared_error(target, y_pred_lr)
r2_lr = r2_score(target, y_pred_lr)
print(f"线性回归模型性能:")
print(f"  均方误差 (MSE): {mse_lr:.6f}")
print(f"  决定系数 (R²): {r2_lr:.6f}")
print(f"  回归系数: {lr_model.coef_}")

# 3. 绘制线性回归降维结果
plt.figure(figsize=(12, 5))

# 子图1: 降维结果与目标值对比
plt.subplot(1, 2, 1)
plt.plot(target, label='目标值', linewidth=2, color='red')
plt.plot(y_pred_lr, label='线性回归预测', linewidth=1.5, color='blue', alpha=0.8)
plt.xlabel('时间步')
plt.ylabel('健康状态值')
plt.title('线性回归降维结果')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=72, color='green', linestyle='--', alpha=0.5)

# 子图2: 预测值与目标值散点图
plt.subplot(1, 2, 2)
plt.scatter(target, y_pred_lr, alpha=0.6, s=20)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='理想线')
plt.xlabel('目标值')
plt.ylabel('预测值')
plt.title(f'线性回归: R² = {r2_lr:.4f}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('线性回归降维结果.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 保存线性回归结果
lr_results = pd.DataFrame({
    '时间步': range(1, time_steps + 1),
    '目标值': target,
    '线性回归降维值': y_pred_lr
})
lr_results.to_csv('线性回归降维结果.csv', index=False)
print("\n线性回归降维结果已保存到 '线性回归降维结果.csv'")
