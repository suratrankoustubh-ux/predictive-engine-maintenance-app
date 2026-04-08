import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

DATA_REPO_ID = "koustubhsuratran/predictive-engine-maintenance-data"
HF_TOKEN     = os.getenv("HF_TOKEN")

MASTER_FOLDER = Path("engine-predictive-maintenance")
DATA_FOLDER   = MASTER_FOLDER / "data"

TARGET_COL    = "engine_condition"
FEATURES      = [
    "engine_rpm", "lub_oil_pressure", "fuel_pressure",
    "coolant_pressure", "lub_oil_temp", "coolant_temp"
]

# Create local folder structure
for sub in ["raw", "processed", "splits"]:
    (DATA_FOLDER / sub).mkdir(parents=True, exist_ok=True)

# Load raw dataset from Hugging Face
print("Loading raw dataset from Hugging Face...")
dataset = load_dataset(DATA_REPO_ID, data_files="data/raw/engine_data.csv")
pm_df   = dataset["train"].to_pandas()
print(f"  Shape: {pm_df.shape}")
print(f"  Columns: {pm_df.columns.tolist()}")

# Clean the data
print("\n Cleaning data...")

# Rename columns to snake_case
pm_df.columns = (
    pm_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Drop rows with missing values
before = len(pm_df)
pm_df.dropna(inplace=True)
print(f"  Dropped {before - len(pm_df)} rows with nulls.")

# Drop duplicate rows
before = len(pm_df)
pm_df.drop_duplicates(inplace=True)
print(f"  Dropped {before - len(pm_df)} duplicate rows.")

# Keep only required columns
pm_df_cleaned = pm_df[FEATURES + [TARGET_COL]].copy()
print(f"  Cleaned shape: {pm_df_cleaned.shape}")
print(f"  Target distribution:\n{pm_df_cleaned[TARGET_COL].value_counts().to_string()}")

# Split into train / test and save locally ──────────────────────────
print("\n[Step 3] Splitting into train / test sets...")
X = pm_df_cleaned.drop(TARGET_COL, axis=1)
y = pm_df_cleaned[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df          = X_train.copy()
train_df[TARGET_COL] = y_train

test_df           = X_test.copy()
test_df[TARGET_COL]  = y_test

print(f"  Train shape: {train_df.shape} | Test shape: {test_df.shape}")

# Save locally
pm_df_cleaned.to_csv(DATA_FOLDER / "processed" / "pm_df_cleaned.csv", index=False)
train_df.to_csv(DATA_FOLDER / "splits" / "train_data.csv", index=False)
test_df.to_csv(DATA_FOLDER / "splits"  / "test_data.csv",  index=False)
print("  Files saved locally.")

# Upload back to Hugging Face
print("\n Uploading datasets to Hugging Face...")
api = HfApi()

uploads = [
    (DATA_FOLDER / "processed" / "pm_df_cleaned.csv", "data/processed/pm_df_cleaned.csv"),
    (DATA_FOLDER / "splits"    / "train_data.csv",    "data/splits/train_data.csv"),
    (DATA_FOLDER / "splits"    / "test_data.csv",     "data/splits/test_data.csv"),
]

for local_path, repo_path in uploads:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"  Uploaded: {repo_path}")

print("\n prepare_data.py Done.")
