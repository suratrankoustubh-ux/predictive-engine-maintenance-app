import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "koustubhsuratran/predictive-engine-maintenance-app"
HF_TOKEN      = os.getenv("HF_TOKEN")

DEPLOYMENT_FILES = [
    ("Dockerfile",       "Dockerfile"),
    ("requirements.txt", "requirements.txt"),
    ("app/app.py",       "app/app.py"),
]

def deploy():
    api = HfApi()
    print(f"\n Creating or verifying Space: {SPACE_REPO_ID}")
    create_repo(repo_id=SPACE_REPO_ID, repo_type="space",
                space_sdk="docker", token=HF_TOKEN, exist_ok=True)

    print("Uploading deployment files...")
    for local_path, repo_path in DEPLOYMENT_FILES:
        if not Path(local_path).exists():
            print(f"  [SKIP] {local_path} not found.")
            continue
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"{local_path} → {repo_path}")

    print(f"\n Space live at: https://huggingface.co/spaces/{SPACE_REPO_ID}")

if __name__ == "__main__":
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable not set.")
    deploy()