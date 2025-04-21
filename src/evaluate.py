import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, roc_auc_score
from model_deepr import Deepr
from collections import Counter
from pyhealth.datasets import MIMIC3Dataset

# Step 1: Load saved val_samples
import pickle
with open("data/val_samples.pkl", "rb") as f:
    samples = pickle.load(f)

# Step 2: Load vocab from training
with open("data/code_to_idx.pkl", "rb") as f:
    code_to_idx = pickle.load(f)

class DeeprDataset(torch.utils.data.Dataset):
    def __init__(self, samples, code_to_idx):
        self.samples = samples
        self.code_to_idx = code_to_idx
        self.unk = code_to_idx.get("<unk>", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = [self.code_to_idx.get(code, self.unk) for code in sample["conditions"]]
        return torch.tensor(input_ids), torch.tensor(sample["label"], dtype=torch.float)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)
    return inputs_padded, labels_tensor

# Step 3: Create DataLoader directly from val_samples
val_dataset = DeeprDataset(samples, code_to_idx)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Step 4: Load model and evaluate
model = Deepr(vocab_size=len(code_to_idx))
model.load_state_dict(torch.load("models/deepr_model.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation AUROC: {auc:.4f}")

# Step 5: Plot confusion matrix and ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
