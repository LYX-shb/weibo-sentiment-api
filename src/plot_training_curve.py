import pandas as pd
import matplotlib.pyplot as plt

# 读取训练历史
history_std = pd.read_csv('../results/history_binary_std.csv')
history_focal = pd.read_csv('../results/history_binary_focal.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss曲线
axes[0].plot(history_std['epoch'], history_std['loss'], label='标准交叉熵', marker='o')
axes[0].plot(history_focal['epoch'], history_focal['loss'], label='Focal Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失对比')
axes[0].legend()
axes[0].grid(True)

# F1曲线
axes[1].plot(history_std['epoch'], history_std['f1'], label='标准交叉熵', marker='o')
axes[1].plot(history_focal['epoch'], history_focal['f1'], label='Focal Loss', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('验证F1对比')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('../results/training_curve_comparison.png', dpi=150)
print("训练曲线对比图已保存")