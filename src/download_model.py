import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer, BertForSequenceClassification

# 下载到本地
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6)

# 保存到本地
tokenizer.save_pretrained('../bert-base-chinese')
model.save_pretrained('../bert-base-chinese')
print("模型已保存到 ../bert-base-chinese")