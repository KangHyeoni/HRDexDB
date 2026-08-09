#!/usr/bin/env python3
"""Download one HRDexDB episode without scanning the entire dataset."""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ID = "HRDexDB/HRDexDB"
TREE_API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE_API = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
USER_AGENT = "HRDexDB-episode-downloader/1.0"


def request_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return json.load(response)


def episode_files(episode_path, include_video):
    tree_url = f"{TREE_API}/{quote(episode_path, safe='/')}?recursive=true&expand=false"
    entries = request_json(tree_url)
    video_prefix = f"{episode_path}/vid/"
    return [
        entry["path"]
        for entry in entries
        if entry.get("type") == "file"
        and (include_video or not entry.get("path", "").startswith(video_prefix))
    ]


def download_file(file_path, local_dir):
    destination = local_dir / file_path
    if destination.is_file() and destination.stat().st_size > 0:
        return f"{file_path} (already exists)"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    download_url = f"{RESOLVE_API}/{quote(file_path, safe='/')}?download=true"
    request = Request(download_url, headers={"User-Agent": USER_AGENT})

    with urlopen(request) as response, open(temporary, "wb") as output:
        shutil.copyfileobj(response, output)
    os.replace(temporary, destination)
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Download one HRDexDB episode in parallel.")
    parser.add_argument("--hand", required=True, help="Embodiment folder, e.g. allegro_v5")
    parser.add_argument("--object", dest="object_id", required=True, help="Object folder, e.g. instant_rice")
    parser.add_argument("--episode", required=True, help="Episode folder, e.g. 0")
    parser.add_argument("--local-dir", default="HRDexDB", help="Destination root (default: HRDexDB)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default: 8)")
    parser.add_argument("--full", action="store_true", help="Include the vid directory")
    args = parser.parse_args()

    episode_path = "/".join([args.hand, args.object_id, str(args.episode)])
    files = episode_files(episode_path, args.full)
    if not files:
        raise RuntimeError(f"No files found for {episode_path}")

    local_dir = Path(args.local_dir)
    print(f"Downloading {len(files)} files from {episode_path} with {args.workers} workers...")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(download_file, path, local_dir): path for path in files}
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                future.result()
                print(f"[{index}/{len(files)}] {path}")
            except Exception as error:
                failures.append((path, error))
                print(f"[{index}/{len(files)}] failed: {path}: {error}", file=sys.stderr)

    if failures:
        raise RuntimeError(f"Failed to download {len(failures)} file(s)")
    print("Download complete.")


if __name__ == "__main__":
    main()
