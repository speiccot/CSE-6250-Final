# Reproducing Deepr: A Convolutional Net for Medical Records

This project is a reproducibility study of the paper:

> Nguyen, P., Tran, T., Wickramasinghe, N., & Venkatesh, S. (2017). **Deepr: A Convolutional Net for Medical Records**. *IEEE Journal of Biomedical and Health Informatics*, 21(1), 22–30. https://doi.org/10.1109/JBHI.2016.2633963

We implement the Deepr model from scratch using PyTorch to predict 30-day hospital readmission based on MIMIC-III data.

---

## 🔧 Project Structure

```
├── data/                   # Folder for MIMIC-III data and preprocessed files
│   ├── mimiciii/           # Raw data tables
│   ├── code_to_idx.pkl     # Saved vocabulary
│   └── val_samples.pkl     # Validation split
├── models/                 # Trained model weights
│   └── deepr_model.pt
├── src/
│   ├── preprocess.py       # Data preprocessing and label generation
│   ├── train.py            # Model training
│   ├── evaluate.py         # Evaluation and visualization
│   └── model_deepr.py      # Deepr model architecture
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/speiccot/CSE-6250.git
   cd CSE-6250
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv deepr-env
   source deepr-env/bin/activate  # On Windows: .\deepr-env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the MIMIC-III data**:
   Place raw CSV files under `data/mimiciii/`. The required tables include:
   - `PATIENTS.csv`
   - `ADMISSIONS.csv`
   - `DIAGNOSES_ICD.csv`

---

## 🏃‍♂️ Running the Project

- **Preprocess data**:
  ```bash
  python src/preprocess.py
  ```

- **Train model**:
  ```bash
  python src/train.py
  ```

- **Evaluate model**:
  ```bash
  python src/evaluate.py
  ```

---

## 📈 Results

- **Accuracy**: 0.84  
- **AUROC**: 0.81  
- Visuals include confusion matrix and ROC curve (see `evaluate.py`)

---

## 📄 Report & Presentation

- Final paper and slides included in the repo root.
- [Presentation Video Link](#) ← Replace with YouTube/Drive link.

---

## 📜 License

This project is for educational purposes only. MIMIC-III is licensed for academic use. Please respect the data use agreement.
