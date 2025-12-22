# 1. 综合对比分析
comparison_df = pd.DataFrame({
    '时间步': range(1, time_steps + 1),
    '目标值': target,
    '线性回归': y_pred_lr,
    '神经网络': y_pred_dnn
})

# 2. 计算不同阶段的误差
healthy_phase = comparison_df.iloc[:72]
degrading_phase = comparison_df.iloc[72:]

print("\n" + "="*60)
print("两种降维方法性能对比")
print("="*60)

print(f"\n整体性能:")
print(f"{'方法':<15} {'MSE':<15} {'R²':<15}")
print(f"{'-'*45}")
print(f"{'线性回归':<15} {mse_lr:<15.6f} {r2_lr:<15.6f}")
print(f"{'神经网络':<15} {mse_dnn:<15.6f} {r2_dnn:<15.6f}")

print(f"\n健康阶段 (1-72时间步) 性能:")
mse_lr_healthy = mean_squared_error(healthy_phase['目标值'], healthy_phase['线性回归'])
mse_dnn_healthy = mean_squared_error(healthy_phase['目标值'], healthy_phase['神经网络'])
print(f"  线性回归 MSE: {mse_lr_healthy:.6f}")
print(f"  神经网络 MSE: {mse_dnn_healthy:.6f}")

print(f"\n退化阶段 (73-192时间步) 性能:")
mse_lr_degrading = mean_squared_error(degrading_phase['目标值'], degrading_phase['线性回归'])
mse_dnn_degrading = mean_squared_error(degrading_phase['目标值'], degrading_phase['神经网络'])
print(f"  线性回归 MSE: {mse_lr_degrading:.6f}")
print(f"  神经网络 MSE: {mse_dnn_degrading:.6f}")

# 3. 绘制最终对比图
plt.figure(figsize=(14, 8))

# 子图1: 两种方法对比
plt.subplot(2, 2, 1)
plt.plot(comparison_df['目标值'], label='目标值', linewidth=3, color='red', alpha=0.5)
plt.plot(comparison_df['线性回归'], label='线性回归', linewidth=1.5, color='blue')
plt.plot(comparison_df['神经网络'], label='神经网络', linewidth=1.5, color='purple')
plt.xlabel('时间步', fontsize=12)
plt.ylabel('健康状态值', fontsize=12)
plt.title('降维方法对比: 线性回归 vs 神经网络', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axvline(x=72, color='green', linestyle='--', alpha=0.7, label='退化开始')

# 子图2: 误差对比
plt.subplot(2, 2, 2)
errors_lr = np.abs(comparison_df['目标值'] - comparison_df['线性回归'])
errors_dnn = np.abs(comparison_df['目标值'] - comparison_df['神经网络'])
plt.plot(errors_lr, label='线性回归误差', linewidth=1.5, color='blue', alpha=0.7)
plt.plot(errors_dnn, label='神经网络误差', linewidth=1.5, color='purple', alpha=0.7)
plt.xlabel('时间步', fontsize=12)
plt.ylabel('绝对误差', fontsize=12)
plt.title('两种方法的预测误差对比', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图3: 误差分布直方图
plt.subplot(2, 2, 3)
plt.hist(errors_lr, bins=30, alpha=0.5, label='线性回归', color='blue')
plt.hist(errors_dnn, bins=30, alpha=0.5, label='神经网络', color='purple')
plt.xlabel('绝对误差', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('误差分布对比', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图4: 性能指标对比
plt.subplot(2, 2, 4)
methods = ['线性回归', '神经网络']
mse_values = [mse_lr, mse_dnn]
r2_values = [r2_lr, r2_dnn]

x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, mse_values, width, label='MSE', color='lightblue')
plt.bar(x + width/2, r2_values, width, label='R²', color='lightcoral')
plt.xlabel('方法', fontsize=12)
plt.ylabel('指标值', fontsize=12)
plt.title('性能指标对比', fontsize=14, fontweight='bold')
plt.xticks(x, methods)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('降维方法综合对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 保存所有结果
print("\n" + "="*60)
print("结果文件汇总")
print("="*60)
print("1. '原始数据特征图.png' - 原始12维特征可视化")
print("2. '健康状态目标变量.png' - 目标变量定义")
print("3. '线性回归降维结果.png' - 线性回归降维结果")
print("4. '神经网络降维结果.png' - 神经网络降维结果及训练过程")
print("5. '降维方法综合对比.png' - 两种方法综合对比")
print("6. '线性回归降维结果.csv' - 线性回归数值结果")
print("7. '神经网络降维结果.csv' - 神经网络数值结果")
print("\n分析完成！")