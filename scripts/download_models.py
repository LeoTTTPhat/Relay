"""Cache the model IDs referenced by the experiment sources."""
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-fr",
]

if __name__ == "__main__":
    for model in MODELS:
        print("Caching", model, flush=True)
        snapshot_download(repo_id=model)
