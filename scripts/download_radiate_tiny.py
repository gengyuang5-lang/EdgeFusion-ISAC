from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.dropbox.com/s/dh361agxhn42ywm/tiny_foggy.zip?dl=1"
MAX_DOWNLOAD_MB = 850


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "radiate_tiny_foggy"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "tiny_foggy.zip"
    if not zip_path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) / (1024 * 1024) > MAX_DOWNLOAD_MB:
                raise SystemExit("Download exceeds local budget.")
            with zip_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    extract_dir = raw_dir / "extracted"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    print(f"RADIATE tiny sample ready at {extract_dir}")


if __name__ == "__main__":
    main()
