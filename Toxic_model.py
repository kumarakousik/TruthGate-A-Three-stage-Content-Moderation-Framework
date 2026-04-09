import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class RoBERTa_Toxic_CNN_NER(nn.Module):

    def __init__(self, num_labels=6, dropout=0.3, model_name="unitary/unbiased-toxic-roberta"):
        self.LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        super().__init__()

        # RoBERTa backbone (pre-trained on toxic data)
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 for base

        # CNN layers (n-gram feature extraction)
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_size, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(hidden_size, 128, kernel_size=4)

        # Dropout layers
        self.dropout_roberta = nn.Dropout(dropout)
        self.dropout_cnn = nn.Dropout(dropout)
        self.dropout_mlp1 = nn.Dropout(dropout)
        self.dropout_mlp2 = nn.Dropout(dropout)

        # MLP Classifier (with intermediate layers)
        cnn_output_size = 128 * 3   # 384
        ner_feature_size = 2
        combined_size = cnn_output_size + ner_feature_size  # 386

        self.mlp = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),

            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, has_person, has_org):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        x = self.dropout_roberta(x)
        x = x.permute(0, 2, 1)

        x1 = torch.relu(self.conv1(x))
        x1 = torch.max(x1, dim=2)[0]   # (batch, 128)
        x2 = torch.relu(self.conv2(x))
        x2 = torch.max(x2, dim=2)[0]   # (batch, 128)
        x3 = torch.relu(self.conv3(x))
        x3 = torch.max(x3, dim=2)[0]   # (batch, 128)

        # Concatenate CNN features
        cnn_features = torch.cat([x1, x2, x3], dim=1)  # (batch, 384)
        cnn_features = self.dropout_cnn(cnn_features)

        # Prepare NER features
        if has_person.dim() == 1:
            has_person = has_person.unsqueeze(1)
        if has_org.dim() == 1:
            has_org = has_org.unsqueeze(1)

        ner_features = torch.cat([has_person, has_org], dim=1)  # (batch, 2)

        # Concatenate CNN + NER features
        combined = torch.cat([cnn_features, ner_features], dim=1)  # (batch, 386)

        # MLP classification
        logits = self.mlp(combined)  # (batch, num_labels)

        return logits

    
    def preprocess_text(self, text: str, max_len=128):
        tokenizer = RobertaTokenizer.from_pretrained("unitary/unbiased-toxic-roberta")
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return encoding

    

    def load_model(self, path: str = "RoBERTa_Toxic_CNN_NER_final.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)

        # 🔥 THIS IS THE FIX
        self.load_state_dict(checkpoint["model_state_dict"])

        self.to(device)
        self.eval()
        return self
    
    
    def predict_single(self, encoding, device, threshold=0.5, has_person=0, has_org=0):
        self.eval()

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Convert has_person and has_org to tensors
        has_person_tensor = torch.tensor([[float(has_person)]], dtype=torch.float32).to(device)
        has_org_tensor = torch.tensor([[float(has_org)]], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self(input_ids, attention_mask, has_person_tensor, has_org_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

        preds = preds.squeeze(0).cpu().numpy()
        return preds, probs
    
toxic_model = RoBERTa_Toxic_CNN_NER()