# # src/train.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torch.nn.utils.rnn import pad_sequence
# from collections import Counter

# from pyhealth.datasets import MIMIC3Dataset
# from pyhealth.tasks import readmission_prediction_mimic3_fn

# from model_deepr import Deepr

# # Step 1: Load the real MIMIC-III data (from your local machine)
# dataset = MIMIC3Dataset(
#     root="data/mimiciii",
#     tables=["DIAGNOSES_ICD"],
#     dev=True
# )
# # Step 2: Create labeled prediction task (manual version)
# from datetime import timedelta

# samples = []

# for patient in dataset.patients.values():
#     visits = sorted(patient.visits.values(), key=lambda v: v.encounter_time)
#     for i in range(len(visits) - 1):
#         visit = visits[i]
#         next_visit = visits[i + 1]
#         if visit.encounter_time is None or next_visit.encounter_time is None:
#             continue
#         time_diff = (next_visit.encounter_time - visit.encounter_time).days
#         label = 1 if time_diff <= 30 else 0

#         icd_codes = visit.get_code_list(table="DIAGNOSES_ICD")
#         if len(icd_codes) == 0:
#             continue

#         samples.append({
#             "conditions": icd_codes,
#             "label": label,
#         })

# # Step 3: Build vocabulary from diagnosis codes
# def build_vocab(samples, min_freq=1):
#     counter = Counter(code for sample in samples for code in sample["conditions"])
#     vocab = {"<pad>": 0, "<unk>": 1}
#     for code, freq in counter.items():
#         if freq >= min_freq:
#             vocab[code] = len(vocab)
#     return vocab

# code_to_idx = build_vocab(samples)

# # Step 4: Define Dataset class
# class DeeprDataset(Dataset):
#     def __init__(self, samples, code_to_idx):
#         self.samples = samples
#         self.code_to_idx = code_to_idx
#         self.unk = self.code_to_idx.get("<unk>", 1)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         code_seq = sample["conditions"]
#         label = sample["label"]
#         input_ids = [self.code_to_idx.get(code, self.unk) for code in code_seq]
#         return torch.tensor(input_ids), torch.tensor(label, dtype=torch.float)

# # Step 5: Collate function for DataLoader
# def collate_fn(batch):
#     inputs, labels = zip(*batch)
#     inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
#     labels_tensor = torch.stack(labels)
#     return inputs_padded, labels_tensor

# # Step 6: Initialize DataLoader
# dataset = DeeprDataset(samples, code_to_idx)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# # Step 7: Initialize model, loss, optimizer
# model = Deepr(vocab_size=len(code_to_idx)).to("cuda" if torch.cuda.is_available() else "cpu")
# device = next(model.parameters()).device

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Step 8: Train loop
# for epoch in range(5):
#     model.train()
#     total_loss = 0
#     for batch in dataloader:
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs).squeeze(1)  # [batch_size]
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# # Step 9: Save the trained model
# model_path = "model\deepr_model.pt"
# torch.save(model.state_dict(), model_path)
# print(f"Trained model saved to {model_path}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
from datetime import timedelta

from pyhealth.datasets import MIMIC3Dataset
from model_deepr import Deepr

# Step 1: Load MIMIC-III dataset
dataset = MIMIC3Dataset(
    root="data/mimiciii",
    tables=["DIAGNOSES_ICD"],
    dev=True
)

# Step 2: Create labeled samples for readmission prediction
samples = []
for patient in dataset.patients.values():
    visits = sorted(patient.visits.values(), key=lambda v: v.encounter_time)
    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]
        if visit.encounter_time is None or next_visit.encounter_time is None:
            continue
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        label = 1 if time_diff <= 30 else 0

        icd_codes = visit.get_code_list(table="DIAGNOSES_ICD")
        if len(icd_codes) == 0:
            continue

        samples.append({
            "conditions": icd_codes,
            "label": label,
        })

# Step 3: Split into train and validation sets
train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

# Save val_samples for evaluation
with open("data/val_samples.pkl", "wb") as f:
    pickle.dump(val_samples, f)

# Step 4: Build vocab
def build_vocab(samples, min_freq=1):
    counter = Counter(code for sample in samples for code in sample["conditions"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for code, freq in counter.items():
        if freq >= min_freq:
            vocab[code] = len(vocab)
    return vocab

code_to_idx = build_vocab(train_samples)

# Step 5: Dataset and DataLoader
class DeeprDataset(Dataset):
    def __init__(self, samples, code_to_idx):
        self.samples = samples
        self.code_to_idx = code_to_idx
        self.unk = self.code_to_idx.get("<unk>", 1)

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

train_dataset = DeeprDataset(train_samples, code_to_idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Step 6: Model, optimizer, and training
model = Deepr(vocab_size=len(code_to_idx)).to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Step 7: Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Step 8: Save model
model_path = "models/deepr_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")
# Save vocab used for training
with open("data/code_to_idx.pkl", "wb") as f:
    pickle.dump(code_to_idx, f)