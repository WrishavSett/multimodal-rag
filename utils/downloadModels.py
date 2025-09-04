from huggingface_hub import snapshot_download

# Choose your target directories
dir = "./models/<model-name>"

# Download full repos into those directories
snapshot_download(repo_id="llava-hf/<model-name>", local_dir=dir)