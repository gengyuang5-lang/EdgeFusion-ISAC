# Local Training Plan

Target machine: RTX 3060 Laptop, 6GB VRAM.

Storage cap: keep the full project below 1.5 GiB.

Recommended runs:

- Quick check: 200-500 samples, 3 epochs, 5-15 minutes.
- Proof of concept: 1k-2k samples, 10-15 epochs, 45-120 minutes.
- Report run: 2k-5k samples, 20-30 epochs, 2-5 hours.

Use sample/tiny splits only. Prefer cached features over raw image training.
