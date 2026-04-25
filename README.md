# EdgeFusion-ISAC

Task-oriented multi-modal semantic fusion for Integrated Sensing and Communication (ISAC) under a small local data budget.

This repository contains the code-facing proof-of-concept for the research proposal `Smart Connectivity and Fusion Research Proposal.docx`. The project studies an edge-oriented ISAC semantic fusion architecture, but the current evidence is intentionally framed as small-scale and preliminary.

## What Is Included

- RADIATE tiny foggy sample experiment for radar-camera perception sanity checking.
- DeepMIMO ASU Campus 3.5GHz experiment for channel-aware semantic slicing proxy evaluation.
- Local storage budget tools for keeping the project under 1.5 GiB.
- Conservative experiment reports that distinguish supported trends from unsupported claims.

## Current Evidence

The current small-scale public-data results support only part of the proposal:

- DeepMIMO: channel-aware adaptive slicing reduces the SER proxy from `0.3904` to `0.3268`, a `16.28%` relative reduction.
- RADIATE tiny foggy: naive radar-camera fusion does **not** outperform camera-only object-count regression on this tiny sample.
- RoI extraction, CRB-constrained beamforming, and full MoE superiority remain design objectives requiring larger-scale validation.

The reports are available at:

- `outputs/reports/deepmimo_tiny_channel_report.json`
- `outputs/reports/radiate_tiny_report.json`

## Limitations

The DeepMIMO result uses a SER proxy derived from ray-tracing channel quality, not an over-the-air measured semantic error rate. The RADIATE result uses only the official tiny foggy sample with 18 aligned radar-camera frames, so it should not be treated as statistically strong evidence.

In its current state, the project supports this cautious claim:

> Small-scale public-data experiments suggest that channel-aware semantic slicing is promising, while multi-modal fusion requires further validation on larger synchronized datasets.

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

## Storage Budget

The local project is designed to stay below `1.5 GiB`.

```powershell
python scripts\check_storage_budget.py
```
