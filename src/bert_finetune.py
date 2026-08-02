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

# ========== 配置（GPU版 RTX 4060） ==========
DATA_DIR = '../data/'
MODEL_NAME = '../bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 标签映射
label_to_num = {'happy': 0, 'sad': 1, 'angry': 2, 'surprise': 3, 'fear': 4, 'neutral': 5}
num_to_label = {v: k for k, v in label_to_num.items()}

# 加载所有数据
usual_train = pd.read_csv(f'{DATA_DIR}usual_train.csv')
usual_eval = pd.read_csv(f'{DATA_DIR}usual_eval_labeled.csv')
usual_test = pd.read_csv(f'{DATA_DIR}usual_test_labeled.csv')
virus_train = pd.read_csv(f'{DATA_DIR}virus_train.csv')
virus_eval = pd.read_csv(f'{DATA_DIR}virus_eval_labeled.csv')
virus_test = pd.read_csv(f'{DATA_DIR}virus_test_labeled.csv')

# 文字标签转数字
for df in [usual_train, usual_eval, usual_test, virus_train, virus_eval, virus_test]:
    df['label'] = df['情绪标签'].map(label_to_num)

print(f"usual训练集: {len(usual_train)}, 验证集: {len(usual_eval)}, 测试集: {len(usual_test)}")
print(f"virus训练集: {len(virus_train)}, 验证集: {len(virus_eval)}, 测试集: {len(virus_test)}")

# 分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 数据集类
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

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
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

# ========== 训练函数 ==========
def train_model(train_df, eval_df, test_df, use_focal=False, save_name='best_model.pt'):
    print(f"\n{'='*60}")
    print(f"训练 {'BERT+FocalLoss' if use_focal else 'BERT(标准)'}")
    print(f"训练集: {len(train_df)}条, 测试集: {len(test_df)}条")
    print(f"{'='*60}")
    
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = WeiboDataset(train_df['文本'], train_df['label'], tokenizer, MAX_LEN)
    eval_dataset = WeiboDataset(eval_df['文本'], eval_df['label'], tokenizer, MAX_LEN)
    test_dataset = WeiboDataset(test_df['文本'], test_df['label'], tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 损失函数
    if use_focal:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(train_df['label'])
        class_weights = compute_class_weight('balanced', classes=classes, y=train_df['label'])
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"类别权重: {class_weights}")
        
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
        
        criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
    else:
        criterion = None
    
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if use_focal:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        eval_acc, eval_f1, _, _ = evaluate(model, eval_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Eval_ACC: {eval_acc:.4f}, Eval_F1: {eval_f1:.4f}")
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), f'../results/{save_name}')
            print(f"  -> 保存最佳模型 (F1={best_f1:.4f})")
    
    # 测试集评估
    model.load_state_dict(torch.load(f'../results/{save_name}'))
    test_acc, test_f1, true_labels, predictions = evaluate(model, test_loader)
    print(f"\n测试集结果: ACC={test_acc:.4f}, F1={test_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(num_to_label.values()))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix')
    plt.savefig(f'../results/confusion_matrix_{save_name.replace(".pt", "")}.png')
    
    return test_acc, test_f1

# ========== 跨领域迁移实验 ==========
if __name__ == '__main__':
    all_results = []
    
    # 实验1: usual训练 -> usual测试（领域内基准）
    print("\n" + "="*70)
    print("实验1: usual训练 -> usual测试（领域内）")
    print("="*70)
    acc1, f1_1 = train_model(usual_train, usual_eval, usual_test, 
                             use_focal=False, save_name='usual_to_usual.pt')
    all_results.append({'实验': 'usual→usual', '训练域': '通用', '测试域': '通用', 
                       'ACC': f'{acc1:.4f}', 'F1': f'{f1_1:.4f}'})
    
    # 实验2: usual训练 -> virus测试（跨领域）
    print("\n" + "="*70)
    print("实验2: usual训练 -> virus测试（跨领域）")
    print("="*70)
    acc2, f1_2 = train_model(usual_train, usual_eval, virus_test, 
                             use_focal=False, save_name='usual_to_virus.pt')
    all_results.append({'实验': 'usual→virus', '训练域': '通用', '测试域': '疫情', 
                       'ACC': f'{acc2:.4f}', 'F1': f'{f1_2:.4f}'})
    
    # 实验3: virus训练 -> virus测试（领域内）
    print("\n" + "="*70)
    print("实验3: virus训练 -> virus测试（领域内）")
    print("="*70)
    acc3, f1_3 = train_model(virus_train, virus_eval, virus_test, 
                             use_focal=False, save_name='virus_to_virus.pt')
    all_results.append({'实验': 'virus→virus', '训练域': '疫情', '测试域': '疫情', 
                       'ACC': f'{acc3:.4f}', 'F1': f'{f1_3:.4f}'})
    
    # 实验4: virus训练 -> usual测试（跨领域）
    print("\n" + "="*70)
    print("实验4: virus训练 -> usual测试（跨领域）")
    print("="*70)
    acc4, f1_4 = train_model(virus_train, virus_eval, usual_test, 
                             use_focal=False, save_name='virus_to_usual.pt')
    all_results.append({'实验': 'virus→usual', '训练域': '疫情', '测试域': '通用', 
                       'ACC': f'{acc4:.4f}', 'F1': f'{f1_4:.4f}'})
    
    # 汇总结果
    print("\n" + "="*70)
    print("跨领域迁移实验结果汇总")
    print("="*70)
    results_df = pd.DataFrame(all_results)
    print(results_df)
    results_df.to_csv('../results/transfer_results.csv', index=False)
    
    # 分析
    print(f"\n领域内平均性能: ACC={(acc1+acc3)/2:.4f}, F1={(f1_1+f1_3)/2:.4f}")
    print(f"跨领域平均性能: ACC={(acc2+acc4)/2:.4f}, F1={(f1_2+f1_4)/2:.4f}")
    print(f"领域迁移损失: ACC下降 {((acc1+acc3)/2-(acc2+acc4)/2)*100:.2f}%, F1下降 {((f1_1+f1_3)/2-(f1_2+f1_4)/2)*100:.2f}%")