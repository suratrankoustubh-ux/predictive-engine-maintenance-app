import os
import json
import joblib
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

DATA_REPO_ID  = "koustubhsuratran/predictive-engine-maintenance-data"
MODEL_REPO_ID = "koustubhsuratran/predictive-engine-maintenance-model"
HF_TOKEN      = os.getenv("HF_TOKEN")

MASTER_FOLDER = Path("engine-predictive-maintenance")
TARGET_COL    = "engine_condition"

# older setup
(MASTER_FOLDER / "models").mkdir(parents=True, exist_ok=True)
(MASTER_FOLDER / "reports" / "experiments").mkdir(parents=True, exist_ok=True)

# Load train / test from Hugging Face
print("\n[Step 1] Loading train and test data from Hugging Face...")
train_ds = load_dataset(DATA_REPO_ID, data_files="data/splits/train_data.csv")
test_ds  = load_dataset(DATA_REPO_ID, data_files="data/splits/test_data.csv")

train_df = train_ds["train"].to_pandas()
test_df  = test_ds["train"].to_pandas()

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

# Define all models and parameter grids
print("\n Defining models and parameter grids...")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

models_and_params = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "max_depth":        [3, 5, 7, 10, None],
            "min_samples_split":[2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion":        ["gini", "entropy"],
        },
        "n_iter": 48,
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 50, 100, 200],
            "max_samples":  [0.5, 0.7, 0.8, 1.0],
            "max_features": [0.5, 0.7, 0.8, 1.0],
        },
        "n_iter": 20,
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "n_estimators":     [50, 100, 200, 300],
            "max_depth":        [5, 10, 15, None],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features":     ["sqrt", "log2", None],
        },
        "n_iter": 30,  # reduced for CI speed
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators":  [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        "n_iter": 6,
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.7, 0.8, 0.9, 1.0],
        },
        "n_iter": 20,  # reduced for CI speed
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
        "params": {
            "n_estimators":     [50, 100, 200],
            "max_depth":        [3, 5, 7],
            "learning_rate":    [0.01, 0.1, 0.2],
            "subsample":        [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
        "n_iter": 20,  # reduced for CI speed
    },
}

# Tune all models with RandomizedSearchCV
print("\n Tuning all models with RandomizedSearchCV...")
results = []

for model_name, mp in models_and_params.items():
    print(f"\n  Tuning: {model_name}...")
    search = RandomizedSearchCV(
        estimator=mp["model"],
        param_distributions=mp["params"],
        n_iter=mp["n_iter"],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"    Best CV F1: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    results.append({
        "Model":           model_name,
        "Best Estimator":  search.best_estimator_,
        "Best Parameters": search.best_params_,
        "Best CV F1 Score":search.best_score_,
    })

# Log all tuned parameters
print("\n Logging tuned parameters...")
tuned_params_log = [
    {
        "Model":           r["Model"],
        "Best Parameters": str(r["Best Parameters"]),
        "Best CV F1 Score":round(r["Best CV F1 Score"], 4),
    }
    for r in results
]
tuned_params_df = pd.DataFrame(tuned_params_log)
log_path = MASTER_FOLDER / "reports" / "experiments" / "tuned_parameters_log.csv"
tuned_params_df.to_csv(log_path, index=False)
print(f"  Saved: {log_path}")
print(tuned_params_df.to_string(index=False))

# Evaluate all models on test set
print("\n[Step 5] Evaluating all models on test set...")
evaluation_results = []

for res in results:
    model_name  = res["Model"]
    best_model  = res["Best Estimator"]
    y_pred      = best_model.predict(X_test)
    y_prob      = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    roc_auc     = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    evaluation_results.append({
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "ROC-AUC":   round(roc_auc, 4) if roc_auc else "N/A",
    })

evaluation_df = pd.DataFrame(evaluation_results).sort_values("F1 Score", ascending=False)
eval_path = MASTER_FOLDER / "reports" / "experiments" / "evaluation_results.csv"
evaluation_df.to_csv(eval_path, index=False)
print(f"\n  Model Comparison:\n{evaluation_df.to_string(index=False)}")

# Register best model to Hugging Face model hub
print("\n Registering best model to Hugging Face model hub...")
best_row        = evaluation_df.iloc[0]
best_model_name = best_row["Model"]
best_model_obj  = next(r["Best Estimator"] for r in results if r["Model"] == best_model_name)
best_params     = next(r["Best Parameters"]  for r in results if r["Model"] == best_model_name)

print(f"\n Best model: {best_model_name}")
print(f"  F1 Score:   {best_row['F1 Score']}")
print(f"  Accuracy:   {best_row['Accuracy']}")
print(f"\n  Classification Report:")
print(classification_report(y_test, best_model_obj.predict(X_test)))

# Save locally
model_path = MASTER_FOLDER / "models" / "best_model.joblib"
joblib.dump(best_model_obj, model_path)
print(f"  Saved locally: {model_path}")


# Upload to HF model hub
if HF_TOKEN:
    api = HfApi()
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

    for local_f, repo_f in [
        (model_path, "models/best_model.joblib"),
        (log_path,   "reports/tuned_parameters_log.csv"),
        (eval_path,  "reports/evaluation_results.csv"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_f),
            path_in_repo=repo_f,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"  Uploaded: {repo_f}")
    print(f"\n  Model hub: https://huggingface.co/{MODEL_REPO_ID}")
else:
    print("  HF_TOKEN not set — skipping upload.")

print("\n train.py Done.")
