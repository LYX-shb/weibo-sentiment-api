import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = '../bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()

def visualize_attention(text, save_path):
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions
    
    # 取最后一层，所有head的平均
    attention = attentions[-1][0].mean(dim=0).numpy()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(attention, cmap='hot', interpolation='nearest')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f'BERT Attention Heatmap\n{text[:30]}...')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"保存: {save_path}")

# 测试样本
test_cases = [
    "今天天气真好，心情特别开心！",
    "气死我了，简直无法忍受！",
    "哈哈哈太搞笑了",
    "这种人真是令人愤怒",
]

for i, text in enumerate(test_cases):
    visualize_attention(text, f'../results/attention_{i}.png')

print("\n所有热力图已保存到 results/")