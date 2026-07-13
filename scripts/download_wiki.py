#!/usr/bin/env python3
"""
Download wiki-18 corpus and index from Hugging Face with memory-efficient streaming.
"""
import os
import gzip
import shutil
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse

def stream_gunzip(in_path, out_path, chunk_size=8192):
    """Gunzip a file in a memory-efficient streaming way."""
    with gzip.open(in_path, 'rb') as f_in:
        # First get uncompressed size for progress bar
        # Read in chunks to decompress
        with open(out_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)

def download_file_with_progress(repo_id, filename, save_dir):
    """Download file from HF with progress bar."""
    print(f"Downloading {filename} from {repo_id}...")
    save_path = os.path.join(save_dir, filename)

    # Use huggingface_hub's built-in download
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=save_dir,
        local_dir_use_symlinks=False,
    )
    print(f"✓ Downloaded {filename} to {save_path}")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Download wiki-18 corpus and index")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save files")
    parser.add_argument("--model", type=str, default="bge", choices=["bge", "e5"], help="Which index to download")

    args = parser.parse_args()

    save_dir = args.save_dir
    model = args.model

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    if model == "bge":
        # Download BGE index (smaller, more memory-efficient)
        print("=== Downloading BGE wiki-18 corpus and index ===")

        # Download corpus (compressed)
        gz_path = download_file_with_progress(
            repo_id="PeterJinGo/wiki-18-corpus",
            filename="wiki-18.jsonl.gz",
            save_dir=save_dir
        )

        # Gunzip it with streaming
        print("Decompressing corpus...")
        corpus_path = os.path.join(save_dir, "corpus.jsonl")
        stream_gunzip(gz_path, corpus_path)

        # Remove compressed file to save space
        os.remove(gz_path)
        print(f"✓ Decompressed to {corpus_path}")

        # Download index parts
        index_dir = os.path.join(save_dir, "index")
        os.makedirs(index_dir, exist_ok=True)

        print("\nDownloading BGE index...")
        for part in ["bge_Flat.index"]:
            download_file_with_progress(
                repo_id="PeterJinGo/wiki-18-bge-index",
                filename=part,
                save_dir=index_dir
            )

        print("\n✅ All files downloaded!")
        print(f"   Corpus: {corpus_path}")
        print(f"   Index: {index_dir}/bge_Flat.index")

    elif model == "e5":
        print("=== Downloading E5 wiki-18 corpus and index ===")

        # Download corpus
        gz_path = download_file_with_progress(
            repo_id="PeterJinGo/wiki-18-corpus",
            filename="wiki-18.jsonl.gz",
            save_dir=save_dir
        )

        # Gunzip
        corpus_path = os.path.join(save_dir, "corpus.jsonl")
        stream_gunzip(gz_path, corpus_path)
        os.remove(gz_path)
        print(f"✓ Decompressed to {corpus_path}")

        # Download E5 index parts
        index_dir = os.path.join(save_dir, "index")
        os.makedirs(index_dir, exist_ok=True)

        print("\nDownloading E5 index (may take a while)...")
        for part in ["part_aa", "part_ab"]:
            download_file_with_progress(
                repo_id="PeterJinGo/wiki-18-e5-index",
                filename=part,
                save_dir=index_dir
            )

        # Combine parts if needed
        combined_index = os.path.join(index_dir, "e5_Flat.index")
        if len(os.listdir(index_dir)) > 2:  # More than just the part files
            print(f"Combining index parts into {combined_index}...")
            with open(combined_index, 'wb') as out_f:
                for part in sorted(os.listdir(index_dir)):
                    part_path = os.path.join(index_dir, part)
                    if os.path.isfile(part_path):
                        with open(part_path, 'rb') as in_f:
                            shutil.copyfileobj(in_f, out_f)
            print(f"✓ Combined index created")

        print("\n✅ All files downloaded!")
        print(f"   Corpus: {corpus_path}")
        print(f"   Index: {index_dir}")

if __name__ == "__main__":
    main()