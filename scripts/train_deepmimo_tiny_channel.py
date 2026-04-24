from __future__ import annotations

import json
import warnings
from pathlib import Path

import deepmimo as dm
import numpy as np
import torch
from torch import nn


def load_quality_dataset(limit: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = dm.load("asu_campus_3p5")
    power = np.asarray(dataset.power, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        best_path_power = np.nanmax(power, axis=1)
    valid = np.isfinite(best_path_power)
    rx_pos = np.asarray(dataset.rx_pos, dtype=np.float32)[valid]
    best_path_power = best_path_power[valid]
    rng = np.random.default_rng(17)
    idx = rng.choice(len(best_path_power), size=min(limit, len(best_path_power)), replace=False)
    x = torch.tensor(rx_pos[idx], dtype=torch.float32)
    y_db = torch.tensor(best_path_power[idx], dtype=torch.float32)
    y = (y_db - y_db.min()) / (y_db.max() - y_db.min()).clamp_min(1e-6)
    return x, y.unsqueeze(-1)


def semantic_error_proxy(channel_quality: torch.Tensor, semantic_ratio: torch.Tensor) -> torch.Tensor:
    effective_quality = (0.2 + 2.8 * channel_quality).clamp_min(1e-4)
    return (torch.exp(-effective_quality * semantic_ratio) + 0.08 * (1.0 - semantic_ratio)).clamp(0.0, 1.0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    x, y = load_quality_dataset()
    torch.manual_seed(19)
    idx = torch.randperm(len(y))
    train_idx, test_idx = idx[: int(len(y) * 0.8)], idx[int(len(y) * 0.8) :]
    mean, std = x[train_idx].mean(0, keepdim=True), x[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    x_train, x_test = (x[train_idx] - mean) / std, (x[test_idx] - mean) / std
    y_train, y_test = y[train_idx], y[test_idx]
    model = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(250):
        loss = nn.functional.mse_loss(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x_test)
        quality_mae = float((pred - y_test).abs().mean())
        quality_rmse = float(torch.sqrt(((pred - y_test) ** 2).mean()))
        fixed_ser = float(semantic_error_proxy(y_test, torch.full_like(y_test, 0.55)).mean())
        adaptive_ratio = (0.25 + 0.65 * pred).clamp(0.25, 0.90)
        adaptive_ser = float(semantic_error_proxy(y_test, adaptive_ratio).mean())
    report = {
        "dataset": "DeepMIMO asu_campus_3p5",
        "task": "predict normalized best-path channel quality from receiver position",
        "samples": int(len(y)),
        "quality_mae": quality_mae,
        "quality_rmse": quality_rmse,
        "fixed_policy_ser_proxy": fixed_ser,
        "adaptive_policy_ser_proxy": adaptive_ser,
        "relative_ser_proxy_reduction_percent": (fixed_ser - adaptive_ser) / fixed_ser * 100.0,
        "supports_channel_aware_semantic_slicing": adaptive_ser < fixed_ser,
        "caveat": "SER is a proxy computed from DeepMIMO ray-tracing channel quality, not an over-the-air measured semantic error rate.",
    }
    out = root / "outputs" / "reports" / "deepmimo_tiny_channel_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
