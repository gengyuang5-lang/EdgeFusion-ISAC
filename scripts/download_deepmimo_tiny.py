from __future__ import annotations

from pathlib import Path

import deepmimo as dm


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw" / "deepmimo"
    output_dir.mkdir(parents=True, exist_ok=True)
    dm.download("asu_campus_3p5", output_dir=str(output_dir))
    print("DeepMIMO asu_campus_3p5 ready")


if __name__ == "__main__":
    main()
