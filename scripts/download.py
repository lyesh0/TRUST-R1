#!/usr/bin/env python3
"""Download and prepare wiki-18 retriever assets for TRUST-R1."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path


DEFAULT_DATA_ROOT = "/root/autodl-fs"
DEFAULT_INDEX_REPO = "PeterJinGo/wiki-18-e5-index"
DEFAULT_CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
INDEX_PARTS = ("part_aa", "part_ab")
INDEX_FILENAME = "e5_Flat.index"
CORPUS_ARCHIVE = "wiki-18.jsonl.gz"
CORPUS_FILENAME = "wiki-18.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare wiki-18 e5 flat retriever assets for TRUST-R1."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SEARCH_DATA_ROOT", DEFAULT_DATA_ROOT),
        help="Root directory for retriever assets. Default: SEARCH_DATA_ROOT or /root/autodl-fs.",
    )
    parser.add_argument(
        "--index-dir",
        help="Directory for wiki-18 e5 index parts and merged e5_Flat.index. Default: DATA_ROOT/indexes/wiki-18.",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Directory for wiki-18 corpus files. Default: DATA_ROOT/data.",
    )
    parser.add_argument(
        "--save_path",
        help="Backward-compatible flat output directory. If set without --index-dir/--corpus-dir, both outputs go here.",
    )
    parser.add_argument(
        "--index-repo-id",
        default=DEFAULT_INDEX_REPO,
        help=f"Hugging Face dataset repo for the e5 index parts. Default: {DEFAULT_INDEX_REPO}.",
    )
    parser.add_argument(
        "--corpus-repo-id",
        default=DEFAULT_CORPUS_REPO,
        help=f"Hugging Face dataset repo for the wiki-18 corpus. Default: {DEFAULT_CORPUS_REPO}.",
    )
    parser.add_argument(
        "--repo_id",
        help="Deprecated alias for --index-repo-id, kept for compatibility with the old script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/rebuild final files even if they already exist.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip downloading and merging the e5 index.",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip downloading and decompressing the wiki-18 corpus.",
    )
    args = parser.parse_args()

    if args.skip_index and args.skip_corpus:
        parser.error("--skip-index and --skip-corpus cannot both be set")

    if args.repo_id:
        print(
            "Warning: --repo_id is deprecated; use --index-repo-id instead.",
            file=sys.stderr,
        )
        args.index_repo_id = args.repo_id

    return args


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_root = Path(args.data_root).expanduser()

    if args.save_path and not args.index_dir and not args.corpus_dir:
        flat_dir = Path(args.save_path).expanduser()
        return data_root, flat_dir, flat_dir

    index_dir = Path(args.index_dir).expanduser() if args.index_dir else data_root / "indexes" / "wiki-18"
    corpus_dir = Path(args.corpus_dir).expanduser() if args.corpus_dir else data_root / "data"
    return data_root, index_dir, corpus_dir


def get_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it in the AutoDL environment, "
            "for example: pip install huggingface_hub"
        ) from exc
    return hf_hub_download


def download_file(repo_id: str, filename: str, local_dir: Path, force: bool) -> Path:
    hf_hub_download = get_hf_hub_download()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
        force_download=force,
    )
    return Path(path)


def merge_index(index_dir: Path, index_repo_id: str, force: bool) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / INDEX_FILENAME

    if index_path.exists() and not force:
        print(f"Index already exists, skipping: {index_path}")
        return index_path

    part_paths = []
    for part in INDEX_PARTS:
        print(f"Downloading index part {part} from {index_repo_id} ...")
        part_paths.append(download_file(index_repo_id, part, index_dir, force))

    missing = [str(path) for path in part_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing downloaded index part(s): {', '.join(missing)}")

    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    print(f"Merging index parts into {index_path} ...")
    with tmp_path.open("wb") as output:
        for part_path in part_paths:
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, output)
    tmp_path.replace(index_path)
    return index_path


def prepare_corpus(corpus_dir: Path, corpus_repo_id: str, force: bool) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / CORPUS_FILENAME

    if corpus_path.exists() and not force:
        print(f"Corpus already exists, skipping: {corpus_path}")
        return corpus_path

    print(f"Downloading corpus archive {CORPUS_ARCHIVE} from {corpus_repo_id} ...")
    archive_path = download_file(corpus_repo_id, CORPUS_ARCHIVE, corpus_dir, force)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing downloaded corpus archive: {archive_path}")

    tmp_path = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    print(f"Decompressing corpus into {corpus_path} ...")
    with gzip.open(archive_path, "rb") as source, tmp_path.open("wb") as output:
        shutil.copyfileobj(source, output)
    tmp_path.replace(corpus_path)
    return corpus_path


def main() -> None:
    args = parse_args()
    data_root, index_dir, corpus_dir = resolve_dirs(args)

    index_path = None
    corpus_path = None

    if not args.skip_index:
        index_path = merge_index(index_dir, args.index_repo_id, args.force)
    if not args.skip_corpus:
        corpus_path = prepare_corpus(corpus_dir, args.corpus_repo_id, args.force)

    print("\nwiki-18 e5 flat assets ready:")
    if index_path:
        print(f"  index:  {index_path}")
    if corpus_path:
        print(f"  corpus: {corpus_path}")
    print("\nLaunch retriever:")
    if index_path and corpus_path:
        print(f"  INDEX_FILE={index_path} CORPUS_FILE={corpus_path} bash retrieval_launch.sh")
    else:
        print(f"  SEARCH_DATA_ROOT={data_root} bash retrieval_launch.sh")


if __name__ == "__main__":
    main()
