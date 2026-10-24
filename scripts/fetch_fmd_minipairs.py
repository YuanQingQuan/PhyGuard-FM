from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "fmd_minipairs_manifest.json"


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def download(url: str, dst: Path, expected_sha: str, expected_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if len(data) != expected_size:
        raise RuntimeError(f"size mismatch for {dst.name}: {len(data)} != {expected_size}")
    got = git_blob_sha(data)
    if got != expected_sha:
        raise RuntimeError(f"Git blob SHA mismatch for {dst.name}: {got} != {expected_sha}")
    dst.write_bytes(data)
    print(f"OK {dst} ({len(data)} bytes, blob {got})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "fmd_minipairs"))
    args = ap.parse_args()
    out = Path(args.out)
    manifest = json.loads(MANIFEST.read_text())
    for pair in manifest["pairs"]:
        pair_dir = out / pair["id"]
        for role in ("raw", "reference", "upstream_denoised"):
            item = pair[role]
            suffix = Path(item["url"]).suffix or ".png"
            download(item["url"], pair_dir / f"{role}{suffix}", item["git_blob_sha"], item["size_bytes"])


if __name__ == "__main__":
    main()
