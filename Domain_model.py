import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class DeBERTaDomainClassifier(nn.Module):

    def __init__(self, num_labels):
        self.LABELS = ['domain_movie', 'domain_politics', 'other_domain']
        super().__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-base")
        hidden_size = self.deberta.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0]  # <s> token
        logits = self.classifier(cls_token)
        return logits
    
    def build_input_text(self, row):
        text = str(row['cleaned_text']) if pd.notna(row['cleaned_text']) else ""

        hashtags = ""
        ht_val = str(row['hashtags']).strip()
        if ht_val and ht_val != 'nan':
            tags = [f"#{t.strip()}" for t in ht_val.split(',') if t.strip()]
            hashtags = " ".join(tags)

        mentions = ""
        mt_val = str(row['mentions']).strip()
        if mt_val and mt_val != 'nan':
            ments = [f"@{m.strip()}" for m in mt_val.split(',') if m.strip()]
            mentions = " ".join(ments)

        combined = " ".join(filter(None, [text, hashtags, mentions]))
        return combined.strip()

    def preprocess_text(self, text, max_len=128):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return encoding

    def load_model(self, path: str = "deberta_domain_classifier.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)
        self.eval()
        return self
    
    def predict_single(self, encoding, device, threshold=0.5):
        self.eval()

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        

        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        preds = preds.squeeze(0).cpu().numpy()

        return preds, probs
    
domain_model = DeBERTaDomainClassifier(num_labels=3)