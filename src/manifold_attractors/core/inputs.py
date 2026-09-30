"""Pinned public inputs; no workstation paths or precomputed paper artifacts."""
import hashlib
from pathlib import Path

MODELS = {'llama3': 'meta-llama/Llama-3.1-8B',
          'olmo3': 'allenai/OLMo-3-1025-7B', 'qwen3': 'Qwen/Qwen3-8B'}
REVISIONS = {'llama3': 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
             'olmo3': 'a81bae42db3975be1671e27b9c9a56da1a9f980f',
             'qwen3': 'b968826d9c46dd6066d109eabc6255188de91218'}
DATASET_REPO = 'NeelNanda/pile-10k'
DATASET_REVISION = '127bfedcd5047750df5ccf3a12979a47bfa0bafa'
DATASET_FILE = 'data/train-00000-of-00001-4746b8785c874cc7.parquet'
DATASET_SHA256 = 'a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(path=None):
    """Download once through the normal HF cache and verify the archived hash."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type='dataset',
                               revision=DATASET_REVISION)
    path = Path(path)
    if digest(path) != DATASET_SHA256:
        raise ValueError(f'{path}: not the pinned paper dataset ({DATASET_SHA256})')
    return path


def model_snapshot(key, revision=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(MODELS[key], revision=revision or REVISIONS[key],
                             allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken'])
