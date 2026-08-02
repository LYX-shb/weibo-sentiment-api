import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import jieba
import warnings
warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
DATA_DIR = '../data/'

# 标签映射
label_to_num = {'happy': 0, 'sad': 1, 'angry': 2, 'surprise': 3, 'fear': 4, 'neutral': 5}

# 加载数据
usual_train = pd.read_csv(f'{DATA_DIR}usual_train.csv')
usual_eval = pd.read_csv(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = pd.read_csv(f'{DATA_DIR}usual_test_labeled.csv')
virus_train = pd.read_csv(f'{DATA_DIR}virus_train.csv')
virus_eval = pd.read_csv(f'{DATA_DIR}virus_eval_labeled.csv')
virus_test = pd.read_csv(f'{DATA_DIR}virus_test_labeled.csv')

# 文字标签转数字
for df in [usual_train, usual_eval, usual_test, virus_train, virus_eval, virus_test]:
    df['label'] = df['情绪标签'].map(label_to_num)

# 中文分词
def tokenize(text):
    return ' '.join(jieba.cut(str(text)))

print("正在进行中文分词...")
for df in [usual_train, usual_eval, usual_test, virus_train, virus_eval, virus_test]:
    df['text_cut'] = df['文本'].apply(tokenize)

# TF-IDF向量化
print("提取TF-IDF特征...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# 合并所有数据拟合vectorizer
all_text = pd.concat([usual_train['text_cut'], virus_train['text_cut']])
vectorizer.fit(all_text)

def run_experiment(train_df, eval_df, test_df, name):
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"{'='*60}")
    
    X_train = vectorizer.transform(train_df['text_cut'])
    X_eval = vectorizer.transform(eval_df['text_cut'])
    X_test = vectorizer.transform(test_df['text_cut'])
    
    y_train = train_df['label']
    y_eval = eval_df['label']
    y_test = test_df['label']
    
    models = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(random_state=42),
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_eval = model.predict(X_eval)
        acc_eval = accuracy_score(y_eval, y_pred_eval)
        f1_eval = f1_score(y_eval, y_pred_eval, average='weighted')
        
        y_pred_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='weighted')
        
        results.append({
            '实验': name,
            '模型': f'TF-IDF+{model_name}',
            'Eval_ACC': f'{acc_eval:.4f}',
            'Eval_F1': f'{f1_eval:.4f}',
            'Test_ACC': f'{acc_test:.4f}',
            'Test_F1': f'{f1_test:.4f}',
        })
        print(f"{model_name}: Test_ACC={acc_test:.4f}, Test_F1={f1_test:.4f}")
    
    return results

# 运行所有实验
all_results = []

# 实验1: usual→usual
all_results.extend(run_experiment(usual_train, usual_eval, usual_test, 'usual→usual'))

# 实验2: usual→virus
all_results.extend(run_experiment(usual_train, usual_eval, virus_test, 'usual→virus'))

# 实验3: virus→virus
all_results.extend(run_experiment(virus_train, virus_eval, virus_test, 'virus→virus'))

# 实验4: virus→usual
all_results.extend(run_experiment(virus_train, virus_eval, usual_test, 'virus→usual'))

# 保存结果
results_df = pd.DataFrame(all_results)
results_df.to_csv('../results/baseline_transfer_results.csv', index=False)
print("\n所有结果:")
print(results_df)