import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

MODEL_ROOT = '../models/'
BASE_ALLOW_PATTERNS = ["*.model", "*.json", "*.bin", "*.py", "*.md", "*.txt"]
LLAMA_ALLOW_PATTERNS = BASE_ALLOW_PATTERNS + ["*.safetensors"]
COMMON_IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot"]

def _local_model_dir(repo_id):
    return MODEL_ROOT + repo_id

def _download_patterns(repo_id):
    if "llama" in repo_id:
        return LLAMA_ALLOW_PATTERNS, ["*.bin"] + COMMON_IGNORE_PATTERNS
    return BASE_ALLOW_PATTERNS, ["*.safetensors"] + COMMON_IGNORE_PATTERNS

def _download_snapshot(repo_id, local_dir, allow_patterns, ignore_patterns):
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

def main(repo_id):
    local_dir = _local_model_dir(repo_id)
    allow_patterns, ignore_patterns = _download_patterns(repo_id)
    _download_snapshot(repo_id, local_dir, allow_patterns, ignore_patterns)

def _parse_args():
    parser = argparse.ArgumentParser(description='Download model from Hugging Face Hub')
    parser.add_argument(
        '--repo_id',
        type=str,
        default='EleutherAI/pythia-160m',
        help='Repository ID of the model to download'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args.repo_id)
