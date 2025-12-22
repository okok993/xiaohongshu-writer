import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 准备PyTorch数据
X_tensor = torch.FloatTensor(X_normalized)
y_tensor = torch.FloatTensor(target).view(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, target, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# 2. 定义神经网络模型 (12-10-8-6-1)
class DNNModel(nn.Module):
    def __init__(self, input_dim=12):
        super(DNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()  # 使用Sigmoid将输出限制在0-1之间
        )

    def forward(self, x):
        return self.network(x)


# 3. 训练神经网络
def train_dnn_model(learning_rate=0.001, epochs=500):
    model = DNNModel(input_dim=12)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    print("开始训练神经网络...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {avg_test_loss:.6f}")

    return model, train_losses, test_losses


# 训练模型
dnn_model, train_losses, test_losses = train_dnn_model(epochs=500)

# 4. 评估神经网络
dnn_model.eval()
with torch.no_grad():
    y_pred_dnn = dnn_model(X_tensor).numpy().flatten()

# 计算评估指标
mse_dnn = mean_squared_error(target, y_pred_dnn)
r2_dnn = r2_score(target, y_pred_dnn)
print(f"\n神经网络模型性能:")
print(f"  均方误差 (MSE): {mse_dnn:.6f}")
print(f"  决定系数 (R²): {r2_dnn:.6f}")

# 5. 绘制神经网络训练过程
plt.figure(figsize=(12, 10))

# 子图1: 训练损失曲线
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='训练损失', linewidth=2)
plt.plot(test_losses, label='测试损失', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('损失值 (MSE)')
plt.title('神经网络训练过程')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2: 降维结果与目标值对比
plt.subplot(2, 2, 2)
plt.plot(target, label='目标值', linewidth=2, color='red')
plt.plot(y_pred_dnn, label='神经网络预测', linewidth=1.5, color='purple', alpha=0.8)
plt.xlabel('时间步')
plt.ylabel('健康状态值')
plt.title('神经网络降维结果')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=72, color='green', linestyle='--', alpha=0.5)

# 子图3: 预测值与目标值散点图
plt.subplot(2, 2, 3)
plt.scatter(target, y_pred_dnn, alpha=0.6, s=20, color='purple')
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='理想线')
plt.xlabel('目标值')
plt.ylabel('预测值')
plt.title(f'神经网络: R² = {r2_dnn:.4f}')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图4: 两种方法对比
plt.subplot(2, 2, 4)
plt.plot(target, label='目标值', linewidth=2, color='red', alpha=0.7)
plt.plot(y_pred_lr, label='线性回归', linewidth=1, color='blue', alpha=0.7)
plt.plot(y_pred_dnn, label='神经网络', linewidth=1, color='purple', alpha=0.7)
plt.xlabel('时间步')
plt.ylabel('健康状态值')
plt.title('两种降维方法对比')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=72, color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('神经网络降维结果.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 保存神经网络结果
dnn_results = pd.DataFrame({
    '时间步': range(1, time_steps + 1),
    '目标值': target,
    '神经网络降维值': y_pred_dnn
})
dnn_results.to_csv('神经网络降维结果.csv', index=False)
print("\n神经网络降维结果已保存到 '神经网络降维结果.csv'")