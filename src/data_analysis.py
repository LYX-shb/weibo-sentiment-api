import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== 路径配置 ==========
DATA_DIR = '../data/'

# 加载所有数据
files = {
    'usual_train': f'{DATA_DIR}usual_train.csv',
    'usual_eval': f'{DATA_DIR}usual_eval_labeled.csv',
    'usual_test': f'{DATA_DIR}usual_test_labeled.csv',
    'virus_train': f'{DATA_DIR}virus_train.csv',
    'virus_eval': f'{DATA_DIR}virus_eval_labeled.csv',
    'virus_test': f'{DATA_DIR}virus_test_labeled.csv',
}

# 读取并合并
all_data = []
for name, path in files.items():
    df = pd.read_csv(path)
    df['dataset'] = name
    all_data.append(df)
    print(f"{name}: {len(df)}条")

df_all = pd.concat(all_data, ignore_index=True)
print(f"\n总样本数: {len(df_all)}")

# 标签映射（文字→数字）
label_to_num = {'happy': 0, 'sad': 1, 'angry': 2, 'surprise': 3, 'fear': 4, 'neutral': 5}
num_to_label = {v: k for k, v in label_to_num.items()}

df_all['label_num'] = df_all['情绪标签'].map(label_to_num)

# 统计分布
print("\n=== 总体分布 ===")
dist = df_all['label_num'].value_counts().sort_index()
for i, count in dist.items():
    print(f"{num_to_label[i]}: {count}")

# 不平衡比例
max_c = dist.max()
min_c = dist.min()
print(f"\n最大类: {max_c}, 最小类: {min_c}")
print(f"不平衡比例: {max_c/min_c:.2f}:1")

# 按主题统计
print("\n=== 通用主题(usual) ===")
usual_df = df_all[df_all['dataset'].str.contains('usual')]
print(usual_df['情绪标签'].value_counts())

print("\n=== 疫情主题(virus) ===")
virus_df = df_all[df_all['dataset'].str.contains('virus')]
print(virus_df['情绪标签'].value_counts())

# 画图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

dist.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('总体分布', fontsize=14)
axes[0].set_xticklabels([num_to_label[i] for i in range(6)], rotation=45)

usual_df['label_num'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('通用主题(usual)', fontsize=14)
axes[1].set_xticklabels([num_to_label[i] for i in range(6)], rotation=45)

virus_df['label_num'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='red', alpha=0.7)
axes[2].set_title('疫情主题(virus)', fontsize=14)
axes[2].set_xticklabels([num_to_label[i] for i in range(6)], rotation=45)

plt.tight_layout()
plt.savefig('../results/label_distribution.png', dpi=150)
print("\n图表已保存: ../results/label_distribution.png")