from huggingface_hub import snapshot_download

# Choose your target directories
dir = "./models/sentence-transformers/all-MiniLM-L6-v2"

# Download full repos into those directories
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=dir)