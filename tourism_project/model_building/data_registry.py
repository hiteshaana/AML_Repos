from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "hiteshsharma/tourism-dataset"

token = os.getenv("HF_TOKENN")

if not token:
    # Instruct the user on how to set the HF_TOKENN in Colab
    raise ValueError("HF_TOKENN environment variable is missing. Please set it using `os.environ['HF_TOKEN'] = 'hf_your_token_here'` in a preceding cell, or use Colab's 'Secrets' feature (recommended for sensitive data).")

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating dataset repository: {repo_id}")

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        token=token
    )

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type="dataset",
    token=token
)

print("Dataset uploaded successfully.")
