# Resume Analyzer - Training Script
# 45 days internship project - deep learning with pytorch

# Step 1: Import Libraries
import json
import re
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
print("Libraries imported!")

# Step 2: Load Dataset
possible_paths = [
    Path("data/UpdatedResumeDataSet.csv"),
    Path("UpdatedResumeDataSet.csv"),
    Path("data/Resume.csv"),
    Path("Resume.csv"),
    Path("archive (7)/Resume/Resume.csv")
]
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

data_path = next((p for p in possible_paths if p.exists()), None)
if data_path is None:
    print("Dataset not found! Place the dataset in the current folder or in data/ with one of these names:")
    for p in possible_paths:
        print(f"  - {p}")
    exit()

print(f"Using dataset file: {data_path}")
df = pd.read_csv(data_path)
print(f"Dataset loaded - {len(df)} resumes, {df['Category'].nunique()} categories")

# Auto-detect resume text column
resume_col = None
for col_name in ['Resume', 'Resume_str', 'resume', 'resume_str', 'Text', 'text']:
    if col_name in df.columns:
        resume_col = col_name
        break
if resume_col is None:
    print(f"Could not find resume text column! Available columns: {list(df.columns)}")
    exit()
print(f"Using column: '{resume_col}'")

# Step 3: Missing Values
print("Missing values:", df.isnull().sum().sum())
df = df.dropna()

# cleaning text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', ' ', str(text))
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df['clean'] = df[resume_col].apply(clean_text)
print("Text cleaned!")

# Step 4: Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
print(f"Encoded {len(le.classes_)} categories")
for i, c in enumerate(le.classes_):
    print(f"  {i} = {c}")

joblib.dump(le, model_dir / "label_encoder.pkl")

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df['clean']).toarray()
y = df['label'].values
print(f"TF-IDF shape: {X.shape}")
joblib.dump(tfidf, model_dir / "tfidf_vectorizer.pkl")

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    batch_size=32
)

# Step 6: Neural Network
class ResumeNet(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResumeNet(X_train.shape[1], len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

print(f"\nTraining on {device}...")
print("=" * 50)

losses, train_accs, test_accs = [], [], []
best_acc = 0

for epoch in range(50):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        loss = criterion(out, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == by).sum().item()
        total += by.size(0)
    scheduler.step()

    losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total * 100)

    if (epoch+1) % 5 == 0 or epoch == 0:
        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                tc += (model(bx).argmax(1) == by).sum().item()
                tt += by.size(0)
        tacc = tc / tt * 100
        test_accs.append(tacc)
        print(f"Epoch {epoch+1}/50 | Loss: {losses[-1]:.4f} | Train: {train_accs[-1]:.1f}% | Test: {tacc:.1f}%")
        if tacc > best_acc:
            best_acc = tacc
            torch.save({'state': model.state_dict(), 'inp': X_train.shape[1],
                         'out': len(le.classes_), 'acc': best_acc, 'epoch': epoch+1},
                        model_dir / "resume_classifier.pth")

print(f"\nBest accuracy: {best_acc:.1f}%")

# Step 7: Prediction & Evaluation
ckpt = torch.load(model_dir / "resume_classifier.pth", weights_only=True)
model.load_state_dict(ckpt['state'])
model.eval()

preds, labels = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        preds.extend(model(bx).argmax(1).cpu().numpy())
        labels.extend(by.numpy())

acc = accuracy_score(labels, preds)
print(f"\nFinal Accuracy: {acc*100:.1f}%")
print(classification_report(labels, preds, target_names=le.classes_, zero_division=0))

# save metrics
json.dump({
    "accuracy": float(acc), "best_epoch": ckpt['epoch'], "num_classes": len(le.classes_),
    "categories": list(le.classes_), "train_losses": [float(x) for x in losses],
    "train_accuracies": [float(x) for x in train_accs],
    "test_accuracies": [float(x) for x in test_accs],
    "confusion_matrix": confusion_matrix(labels, preds).tolist()
}, open(model_dir / "metrics.json", "w"), indent=2)

print("Model and metrics saved! Now run: streamlit run app.py")
