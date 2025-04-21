# src/preprocess.py

import os
import pandas as pd
from pyhealth.datasets import MIMIC3Dataset


def debug_csv_headers():
    """
    Prints column headers of PATIENTS.csv and ADMISSIONS.csv to confirm structure.
    """
    for filename in ["PATIENTS.csv", "ADMISSIONS.csv", "DIAGNOSES_ICD.csv"]:
        path = os.path.join("data/mimiciii", filename)
        print(f"\n🔍 Checking contents of: {os.path.abspath(path)}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, nrows=1)
                print(f"✅ {filename} columns:", df.columns.tolist())
            except Exception as e:
                print(f"❌ Failed to read {filename}:", e)
        else:
            print(f"❌ File not found: {path}")


def patch_mimic_headers(mimic_root):
    """
    Ensures column names in required MIMIC-III files are uppercase.
    """
    files_to_patch = ["PATIENTS.csv", "ADMISSIONS.csv", "DIAGNOSES_ICD.csv"]
    for fname in files_to_patch:
        path = os.path.join(mimic_root, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [col.strip().upper() for col in df.columns]
                df.to_csv(path, index=False)
                print(f"🔧 Patched headers in: {path}")
            except Exception as e:
                print(f"❌ Failed to patch {fname}: {e}")
        else:
            print(f"⚠️ File not found: {path}")


def load_mimic_dataset(mimic_root: str, dev_mode: bool = True):
    """
    Loads the MIMIC-III dataset using PyHealth's MIMIC3Dataset wrapper.

    Parameters:
        mimic_root (str): Path to the MIMIC-III folder (where the CSVs are).
        dev_mode (bool): Load only a small subset of patients if True.

    Returns:
        dataset: Loaded PyHealth MIMIC3Dataset object.
    """
    print(f"\n📂 Loading MIMIC-III data from: {os.path.abspath(mimic_root)}")

    # Fix headers first
    patch_mimic_headers(mimic_root)

    # Load using PyHealth
    dataset = MIMIC3Dataset(
        root=mimic_root,
        tables=["DIAGNOSES_ICD"],  # Do not include ADMISSIONS or PATIENTS
        dev=dev_mode
    )

    print("✅ Dataset loaded successfully.\n")
    dataset.stat()
    return dataset


if __name__ == "__main__":
    debug_csv_headers()
    load_mimic_dataset("data/mimiciii", dev_mode=True)
