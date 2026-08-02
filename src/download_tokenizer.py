from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer.save_pretrained('../bert-base-chinese')
print('tokenizer已保存')