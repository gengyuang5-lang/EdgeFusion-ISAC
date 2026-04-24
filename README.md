# EdgeFusion-ISAC

Task-oriented multi-modal semantic fusion for Integrated Sensing and Communication (ISAC) under a small local data budget.

This repository contains lightweight proof-of-concept experiments for the research proposal:

- RADIATE tiny foggy sample: radar-camera fusion sanity check.
- DeepMIMO ASU Campus 3.5GHz: channel-aware semantic resource slicing proxy.
- Storage budget tools for keeping the project under 1.5 GiB locally.

## Current Evidence

The current small-scale results are intentionally conservative:

- DeepMIMO supports a weak preliminary claim that channel-aware semantic slicing can reduce a SER proxy.
- RADIATE tiny foggy sample does **not** support the stronger claim that radar-camera fusion always outperforms camera-only perception.

See:

- `outputs/reports/deepmimo_tiny_channel_report.json`
- `outputs/reports/radiate_tiny_report.json`

## Reproduce

```powershell
python scripts\download_radiate_tiny.py
python scripts\train_radiate_tiny_poc.py
python -m pip install deepmimo
python scripts\download_deepmimo_tiny.py
python scripts\train_deepmimo_tiny_channel.py
python scripts\check_storage_budget.py
```

Third-party raw datasets are not committed to this public repository. Use the scripts to download the small samples locally.
