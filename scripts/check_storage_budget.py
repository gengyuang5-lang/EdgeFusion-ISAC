from __future__ import annotations

from pathlib import Path

PROJECT_LIMIT_MB = 1536
PATH_LIMITS_MB = {
    "data/raw": 850,
    "data/processed": 420,
    "outputs/checkpoints": 120,
    "outputs/reports": 80,
}


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def mb(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    total = mb(directory_size_bytes(root))
    failed = total > PROJECT_LIMIT_MB
    print(f"Project size: {total:.2f} MB / {PROJECT_LIMIT_MB} MB")
    for relative, limit in PATH_LIMITS_MB.items():
        size = mb(directory_size_bytes(root / relative))
        status = "OK" if size <= limit else "OVER"
        print(f"{relative:<22} {size:>8.2f} MB / {limit:>4} MB  {status}")
        failed = failed or size > limit
    if failed:
        raise SystemExit("Storage budget exceeded.")


if __name__ == "__main__":
    main()
