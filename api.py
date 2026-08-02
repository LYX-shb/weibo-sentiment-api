from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

app = FastAPI(title="微博情感分析推理服务")

class BertBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("./bert-base-chinese")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

tokenizer = BertTokenizer.from_pretrained("./bert-base-chinese")
model = BertBinaryClassifier()
model.load_state_dict(torch.load("./results/binary_focal.pt", map_location="cpu"))
model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=1)[0]
    
    happy_prob = probs[0].item()
    angry_prob = probs[1].item()
    
    if happy_prob > angry_prob:
        label = "happy"
        confidence = happy_prob
    else:
        label = "angry"
        confidence = angry_prob
    
    return {
        "text": request.text,
        "label": label,
        "confidence": round(confidence, 4)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)