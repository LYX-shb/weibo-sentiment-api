import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import jieba
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data/'

def load_binary(path):
    df = pd.read_csv(path)
    df = df[df['情绪标签'].isin(['happy', 'angry'])].copy()
    df['label'] = df['情绪标签'].map({'happy': 0, 'angry': 1})
    return df

# 加载
usual_train = load_binary(f'{DATA_DIR}usual_train.csv')
usual_eval = load_binary(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = load_binary(f'{DATA_DIR}usual_test_labeled.csv')
virus_train = load_binary(f'{DATA_DIR}virus_train.csv')
virus_eval = load_binary(f'{DATA_DIR}virus_eval_labeled.csv')
virus_test = load_binary(f'{DATA_DIR}virus_test_labeled.csv')

# 合并训练集
train_df = pd.concat([usual_train, virus_train])
eval_df = usual_eval
test_df = usual_test

print(f"训练集: {len(train_df)}, happy={sum(train_df['label']==0)}, angry={sum(train_df['label']==1)}")

# 分词
def tokenize(text):
    return ' '.join(jieba.cut(str(text)))

train_df['text_cut'] = train_df['文本'].apply(tokenize)
eval_df['text_cut'] = eval_df['文本'].apply(tokenize)
test_df['text_cut'] = test_df['文本'].apply(tokenize)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text_cut'])
X_eval = vectorizer.transform(eval_df['text_cut'])
X_test = vectorizer.transform(test_df['text_cut'])

y_train, y_eval, y_test = train_df['label'], eval_df['label'], test_df['label']

# 训练并评估
results = []
for name, model in [('LR', LogisticRegression(max_iter=1000)), ('SVM', SVC())]:
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    
    results.append({'Model': f'TF-IDF+{name}', 'ACC': f'{acc:.4f}', 'F1': f'{f1:.4f}'})
    print(f"{name}: ACC={acc:.4f}, F1={f1:.4f}")

pd.DataFrame(results).to_csv('../results/binary_baseline.csv', index=False)