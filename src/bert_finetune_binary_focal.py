import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_DIR = '../data/'
MODEL_NAME = '../bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
PATIENCE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

label_map = {'happy': 0, 'angry': 1}

# 加载数据
def load_binary(path):
    df = pd.read_csv(path)
    df = df[df['情绪标签'].isin(['happy', 'angry'])].copy()
    df['label'] = df['情绪标签'].map(label_map)
    return df

usual_train = load_binary(f'{DATA_DIR}usual_train.csv')
usual_eval = load_binary(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = load_binary(f'{DATA_DIR}usual_test_labeled.csv')

print(f"训练集: {len(usual_train)}, happy={sum(usual_train['label']==0)}, angry={sum(usual_train['label']==1)}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ========== 模型 ==========
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
        
        encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length', 
                           truncation=True, return_tensors='pt')
        
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

def train_model(train_df, eval_df, test_df, use_focal=False, save_name='best_model.pt'):
    print(f"\n{'='*60}")
    print(f"训练: {save_name} | {'Focal Loss' if use_focal else '标准交叉熵'}")
    print(f"{'='*60}")
    
    train_loader = DataLoader(BinaryDataset(train_df['文本'], train_df['label']), 
                             batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(BinaryDataset(eval_df['文本'], eval_df['label']), 
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(BinaryDataset(test_df['文本'], test_df['label']), 
                            batch_size=BATCH_SIZE)
    
    model = BertBinaryClassifier(MODEL_NAME)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 损失函数
    if use_focal:
        classes = np.unique(train_df['label'])
        class_weights = compute_class_weight('balanced', classes=classes, y=train_df['label'])
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"类别权重: {class_weights}")
        criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
    else:
        criterion = None
    
    best_f1 = 0
    patience_counter = 0
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if use_focal:
                loss = criterion(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        eval_acc, eval_f1, _, _ = evaluate(model, eval_loader)
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': eval_acc, 'f1': eval_f1})
        
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
    plt.savefig(f'../results/cm_{save_name.replace(".pt", ".png")}')
    plt.close()
    
    # 保存训练曲线
    pd.DataFrame(history).to_csv(f'../results/history_{save_name.replace(".pt", ".csv")}', index=False)

    return test_acc, test_f1

# ========== 对比实验 ==========
if __name__ == '__main__':
    results = []
    
    # 标准交叉熵
    acc_std, f1_std = train_model(usual_train, usual_eval, usual_test, 
                                   use_focal=False, save_name='binary_std.pt')
    results.append({'方法': '标准交叉熵', 'ACC': f'{acc_std:.4f}', 'F1': f'{f1_std:.4f}'})
    
    # Focal Loss
    acc_focal, f1_focal = train_model(usual_train, usual_eval, usual_test, 
                                       use_focal=True, save_name='binary_focal.pt')
    results.append({'方法': 'Focal Loss', 'ACC': f'{acc_focal:.4f}', 'F1': f'{f1_focal:.4f}'})
    
    # 汇总
    print(f"\n{'='*60}")
    print("Focal Loss对比实验结果")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('../results/focal_comparison.csv', index=False)
    
    print(f"\nFocal Loss提升: ACC {(acc_focal-acc_std)*100:.2f}%, F1 {f1_focal-f1_std:.4f}")

