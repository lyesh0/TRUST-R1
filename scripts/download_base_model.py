#!/usr/bin/env python3
"""Download Qwen2.5-3B base model to AutoDL models directory."""
import os
import sys

# Use HF mirror for faster download in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-3B"
LOCAL_DIR = "/root/autodl-tmp/models/Qwen2.5-3B"

print(f"Downloading {MODEL_ID} -> {LOCAL_DIR}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

# Verify
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(LOCAL_DIR, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(LOCAL_DIR, trust_remote_code=True)
print(f"\nDownload complete!")
print(f"  model_type: {cfg.model_type}")
print(f"  has_chat_template: {tok.chat_template is not None}")
print(f"  vocab_size: {tok.vocab_size}")

# Check disk usage
import subprocess
subprocess.run(["du", "-sh", LOCAL_DIR])
