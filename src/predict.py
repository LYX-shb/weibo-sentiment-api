import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

# 配置
MODEL_NAME = '../bert-base-chinese'
MODEL_PATH = '../results/binary_usual_usual.pt'
MAX_LEN = 128

# 二分类标签
num_to_label = {0: 'happy', 1: 'angry'}
label_emoji = {'happy': '😊', 'angry': '😠'}

device = torch.device('cuda')

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 定义模型
class BertBinaryClassifier(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# 加载模型
model = BertBinaryClassifier(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=MAX_LEN, 
                      padding='max_length', truncation=True)
    # 只保留需要的输入
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device)
    }
    
    with torch.no_grad():
        outputs = model(**model_inputs)
        probs = torch.softmax(outputs, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    
    return num_to_label[pred], probs[0].cpu().numpy()

# 测试
test_examples = [
    "今天天气真好，心情特别开心！",
    "气死我了，简直无法忍受！",
    "哈哈哈太搞笑了",
    "这种人真是令人愤怒",
]

print("=" * 50)
print("二分类情感预测演示")
print("=" * 50)

for text in test_examples:
    label, probs = predict(text)
    emoji = label_emoji[label]
    conf = max(probs) * 100
    
    print(f"\n文本: {text}")
    print(f"预测: {emoji} {label} (置信度: {conf:.1f}%)")
    print(f"happy: {probs[0]*100:.1f}%, angry: {probs[1]*100:.1f}%")
    print("-" * 50)