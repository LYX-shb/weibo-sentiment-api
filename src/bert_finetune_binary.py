import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 二分类配置 ==========
DATA_DIR = '../data/'
MODEL_NAME = '../bert-base-chinese'  # 本地路径
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
PATIENCE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 二分类标签
label_map = {'happy': 0, 'angry': 1}
num_to_label = {0: 'happy', 1: 'angry'}

# 加载并过滤二分类数据
def load_binary(path):
    df = pd.read_csv(path)
    df = df[df['情绪标签'].isin(['happy', 'angry'])].copy()
    df['label'] = df['情绪标签'].map(label_map)
    return df

usual_train = load_binary(f'{DATA_DIR}usual_train.csv')
usual_eval = load_binary(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = load_binary(f'{DATA_DIR}usual_test_labeled.csv')
virus_train = load_binary(f'{DATA_DIR}virus_train.csv')
virus_eval = load_binary(f'{DATA_DIR}virus_eval_labeled.csv')
virus_test = load_binary(f'{DATA_DIR}virus_test_labeled.csv')

print(f"usual: 训练{len(usual_train)}, 验证{len(usual_eval)}, 测试{len(usual_test)}")
print(f"virus: 训练{len(virus_train)}, 验证{len(virus_eval)}, 测试{len(virus_test)}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ========== 自定义二分类模型 ==========
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)  # 二分类
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class BinaryDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, loader):
    model.eval()
    preds, true = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs, dim=1)
            
            preds.extend(pred.cpu().numpy())
            true.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    return acc, f1, true, preds

def train_model(train_df, eval_df, test_df, save_name):
    print(f"\n{'='*60}")
    print(f"训练: {save_name} | 训练{len(train_df)}条")
    print(f"{'='*60}")
    
    train_loader = DataLoader(BinaryDataset(train_df['文本'], train_df['label']), 
                             batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(BinaryDataset(eval_df['文本'], eval_df['label']), 
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(BinaryDataset(test_df['文本'], test_df['label']), 
                            batch_size=BATCH_SIZE)
    
    # 使用自定义模型
    model = BertBinaryClassifier(MODEL_NAME)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估
        eval_acc, eval_f1, _, _ = evaluate(model, eval_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Eval_ACC: {eval_acc:.4f}, Eval_F1: {eval_f1:.4f}")
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), f'../results/{save_name}')
            print(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ✗ 未提升 (耐心{patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  ! 早停")
                break
    
    # 测试
    model.load_state_dict(torch.load(f'../results/{save_name}'))
    test_acc, test_f1, true_labels, predictions = evaluate(model, test_loader)
    print(f"\n测试集: ACC={test_acc:.4f}, F1={test_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['happy', 'angry'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix - {save_name}')
    plt.savefig(f'../results/cm_binary_{save_name.replace(".pt", ".png")}')
    plt.close()
    
    return test_acc, test_f1

# ========== 运行4个实验 ==========
if __name__ == '__main__':
    results = []
    
    # 实验1: usual->usual
    acc1, f1_1 = train_model(usual_train, usual_eval, usual_test, 'binary_usual_usual.pt')
    results.append({'实验': 'usual→usual', '训练域': '通用', '测试域': '通用', 'ACC': f'{acc1:.4f}', 'F1': f'{f1_1:.4f}'})
    
    # 实验2: usual->virus
    acc2, f1_2 = train_model(usual_train, usual_eval, virus_test, 'binary_usual_virus.pt')
    results.append({'实验': 'usual→virus', '训练域': '通用', '测试域': '疫情', 'ACC': f'{acc2:.4f}', 'F1': f'{f1_2:.4f}'})
    
    # 实验3: virus->virus
    acc3, f1_3 = train_model(virus_train, virus_eval, virus_test, 'binary_virus_virus.pt')
    results.append({'实验': 'virus→virus', '训练域': '疫情', '测试域': '疫情', 'ACC': f'{acc3:.4f}', 'F1': f'{f1_3:.4f}'})
    
    # 实验4: virus->usual
    acc4, f1_4 = train_model(virus_train, virus_eval, usual_test, 'binary_virus_usual.pt')
    results.append({'实验': 'virus→usual', '训练域': '疫情', '测试域': '通用', 'ACC': f'{acc4:.4f}', 'F1': f'{f1_4:.4f}'})
    
    # 汇总
    print(f"\n{'='*60}")
    print("二分类结果汇总")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('../results/binary_results.csv', index=False)
    
    in_acc = (acc1 + acc3) / 2
    in_f1 = (f1_1 + f1_3) / 2
    cross_acc = (acc2 + acc4) / 2
    cross_f1 = (f1_2 + f1_4) / 2
    
    print(f"\n领域内平均: ACC={in_acc:.4f}, F1={in_f1:.4f}")
    print(f"跨领域平均: ACC={cross_acc:.4f}, F1={cross_f1:.4f}")
    print(f"迁移损失: ACC下降 {(in_acc-cross_acc)*100:.2f}%")