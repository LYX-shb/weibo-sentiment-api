import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 优化配置（目标ACC≥92%） ==========
DATA_DIR = '../data/'
MODEL_NAME = '../bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10          # 增加到10轮
LR = 5e-6            # 更小的学习率
PATIENCE = 3         # 早停耐心值
WEIGHT_DECAY = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 标签映射
label_to_num = {'happy': 0, 'sad': 1, 'angry': 2, 'surprise': 3, 'fear': 4, 'neutral': 5}
num_to_label = {v: k for k, v in label_to_num.items()}

# 加载数据
usual_train = pd.read_csv(f'{DATA_DIR}usual_train.csv')
usual_eval = pd.read_csv(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = pd.read_csv(f'{DATA_DIR}usual_test_labeled.csv')
virus_train = pd.read_csv(f'{DATA_DIR}virus_train.csv')
virus_eval = pd.read_csv(f'{DATA_DIR}virus_eval_labeled.csv')
virus_test = pd.read_csv(f'{DATA_DIR}virus_test_labeled.csv')

for df in [usual_train, usual_eval, usual_test, virus_train, virus_eval, virus_test]:
    df['label'] = df['情绪标签'].map(label_to_num)

# 调试：打印标签分布
print("\n=== 标签分布检查 ===")
print("usual_train:", usual_train['label'].value_counts().sort_index().to_dict())
print("usual_eval:", usual_eval['label'].value_counts().sort_index().to_dict())
print("usual_test:", usual_test['label'].value_counts().sort_index().to_dict())
print("virus_train:", virus_train['label'].value_counts().sort_index().to_dict())
print("virus_eval:", virus_eval['label'].value_counts().sort_index().to_dict())
print("virus_test:", virus_test['label'].value_counts().sort_index().to_dict())

print(f"\nusual: 训练{len(usual_train)}, 验证{len(usual_eval)}, 测试{len(usual_test)}")
print(f"virus: 训练{len(virus_train)}, 验证{len(virus_eval)}, 测试{len(virus_test)}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return acc, f1, true_labels, predictions

def train_model(train_df, eval_df, test_df, save_name='best_model.pt'):
    print(f"\n{'='*60}")
    print(f"训练: {save_name}")
    print(f"训练集: {len(train_df)}条, 验证集: {len(eval_df)}条, 测试集: {len(test_df)}条")
    print(f"{'='*60}")
    
    # 数据集
    train_dataset = WeiboDataset(train_df['文本'], train_df['label'], tokenizer, MAX_LEN)
    eval_dataset = WeiboDataset(eval_df['文本'], eval_df['label'], tokenizer, MAX_LEN)
    test_dataset = WeiboDataset(test_df['文本'], test_df['label'], tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 模型
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    model = model.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # 训练
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估
        eval_acc, eval_f1, _, _ = evaluate(model, eval_loader)
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'eval_acc': eval_acc, 'eval_f1': eval_f1})
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Eval_ACC: {eval_acc:.4f}, Eval_F1: {eval_f1:.4f}")
        
        # 保存最佳
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), f'../results/{save_name}')
            print(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ✗ 未提升 (耐心{patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"  ! 早停 (连续{PATIENCE}轮未提升)")
                break
    
    # 加载最佳模型测试
    print(f"\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load(f'../results/{save_name}'))
    test_acc, test_f1, true_labels, predictions = evaluate(model, test_loader)
    print(f"测试集结果: ACC={test_acc:.4f}, F1={test_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(num_to_label.values()))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix - {save_name}')
    plt.savefig(f'../results/cm_{save_name.replace(".pt", ".png")}')
    plt.close()
    
    # 保存训练历史
    pd.DataFrame(history).to_csv(f'../results/history_{save_name.replace(".pt", ".csv")}', index=False)
    
    return test_acc, test_f1

# ========== 运行实验 ==========
if __name__ == '__main__':
    all_results = []
    
    # 实验1: usual->usual（领域内）
    print("\n" + "="*70)
    print("实验1: usual训练 -> usual测试（领域内基准）")
    print("="*70)
    acc1, f1_1 = train_model(usual_train, usual_eval, usual_test, 'usual_to_usual_v2.pt')
    all_results.append({'实验': 'usual→usual', '训练域': '通用', '测试域': '通用', 'ACC': f'{acc1:.4f}', 'F1': f'{f1_1:.4f}'})
    
    # 实验2: usual->virus（跨领域）
    print("\n" + "="*70)
    print("实验2: usual训练 -> virus测试（跨领域）")
    print("="*70)
    acc2, f1_2 = train_model(usual_train, usual_eval, virus_test, 'usual_to_virus_v2.pt')
    all_results.append({'实验': 'usual→virus', '训练域': '通用', '测试域': '疫情', 'ACC': f'{acc2:.4f}', 'F1': f'{f1_2:.4f}'})
    
    # 实验3: virus->virus（领域内）
    print("\n" + "="*70)
    print("实验3: virus训练 -> virus测试（领域内）")
    print("="*70)
    acc3, f1_3 = train_model(virus_train, virus_eval, virus_test, 'virus_to_virus_v2.pt')
    all_results.append({'实验': 'virus→virus', '训练域': '疫情', '测试域': '疫情', 'ACC': f'{acc3:.4f}', 'F1': f'{f1_3:.4f}'})
    
    # 实验4: virus->usual（跨领域）
    print("\n" + "="*70)
    print("实验4: virus训练 -> usual测试（跨领域）")
    print("="*70)
    acc4, f1_4 = train_model(virus_train, virus_eval, usual_test, 'virus_to_usual_v2.pt')
    all_results.append({'实验': 'virus→usual', '训练域': '疫情', '测试域': '通用', 'ACC': f'{acc4:.4f}', 'F1': f'{f1_4:.4f}'})
    
    # 汇总
    print("\n" + "="*70)
    print("BERT优化后结果汇总")
    print("="*70)
    results_df = pd.DataFrame(all_results)
    print(results_df)
    results_df.to_csv('../results/bert_results_v2.csv', index=False)
    
    # 分析
    in_domain_acc = (acc1 + acc3) / 2
    in_domain_f1 = (f1_1 + f1_3) / 2
    cross_acc = (acc2 + acc4) / 2
    cross_f1 = (f1_2 + f1_4) / 2
    
    print(f"\n领域内平均性能: ACC={in_domain_acc:.4f}, F1={in_domain_f1:.4f}")
    print(f"跨领域平均性能: ACC={cross_acc:.4f}, F1={cross_f1:.4f}")
    print(f"领域迁移损失: ACC下降 {(in_domain_acc-cross_acc)*100:.2f}%, F1下降 {(in_domain_f1-cross_f1)*100:.2f}%")
    
    # 与基线对比
    print(f"\n与传统方法对比:")
    print(f"TF-IDF+SVM领域内平均: ACC~67%, F1~66%")
    print(f"BERT领域内平均: ACC={in_domain_acc*100:.2f}%, F1={in_domain_f1:.4f}")
    print(f"提升幅度: ACC提升 {(in_domain_acc-0.67)*100:.2f}%, F1提升 {in_domain_f1-0.66:.4f}")