from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn


@dataclass(frozen=True)
class Result:
    name: str
    test_mae: float
    test_rmse: float


def load_image_feature(path: Path, size: int = 16) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size))
    pixel_data = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
    return torch.tensor(list(pixel_data), dtype=torch.float32) / 255.0


def frame_object_counts(annotation_path: Path, frame_count: int) -> torch.Tensor:
    objects = json.loads(annotation_path.read_text(encoding="utf-8"))
    counts = torch.zeros(frame_count, dtype=torch.float32)
    for frame_idx in range(frame_count):
        for item in objects:
            bboxes = item.get("bboxes", [])
            if frame_idx < len(bboxes) and bboxes[frame_idx]:
                counts[frame_idx] += 1.0
    return counts


def build_dataset(root: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sample = root / "data" / "raw" / "radiate_tiny_foggy" / "extracted" / "tiny_foggy"
    radar_paths = sorted((sample / "Navtech_Cartesian").glob("*.png"))
    camera_paths = sorted((sample / "zed_left").glob("*.png"))
    frame_count = min(len(radar_paths), len(camera_paths))
    if frame_count < 8:
        raise SystemExit("Not enough aligned frames. Run download_radiate_tiny.py first.")
    radar = torch.stack([load_image_feature(path) for path in radar_paths[:frame_count]])
    camera = torch.stack([load_image_feature(path) for path in camera_paths[:frame_count]])
    return {"camera": camera, "radar": radar, "fusion": torch.cat([camera, radar], dim=-1)}, frame_object_counts(sample / "annotations" / "annotations.json", frame_count)


def train_regressor(name: str, x: torch.Tensor, y: torch.Tensor) -> Result:
    torch.manual_seed(23)
    indices = torch.randperm(len(y))
    split = max(1, int(len(y) * 0.7))
    train_idx, test_idx = indices[:split], indices[split:]
    train = x[train_idx]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train, x_test = (x[train_idx] - mean) / std, (x[test_idx] - mean) / std
    y_train, y_test = y[train_idx].unsqueeze(-1), y[test_idx]
    model = nn.Sequential(nn.Linear(x_train.shape[-1], 32), nn.GELU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(300):
        loss = nn.functional.mse_loss(model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = model(x_test).squeeze(-1).clamp_min(0.0)
        err = pred - y_test
        return Result(name, float(err.abs().mean()), float(torch.sqrt((err**2).mean())))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features, labels = build_dataset(root)
    results = [train_regressor(name, tensor, labels) for name, tensor in features.items()]
    best = min(results, key=lambda item: item.test_mae)
    fusion = next(item for item in results if item.name == "fusion")
    camera = next(item for item in results if item.name == "camera")
    radar = next(item for item in results if item.name == "radar")
    report = {
        "dataset": "RADIATE tiny_foggy official sample",
        "task": "frame-level object-count regression from aligned camera/radar image features",
        "frames": int(len(labels)),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_mean": float(labels.mean()),
        "results": [result.__dict__ for result in results],
        "best_modality": best.name,
        "supports_fusion_on_this_sample": fusion.test_mae < min(camera.test_mae, radar.test_mae),
        "caveat": "This is a tiny single-fog-sequence proof-of-concept, not a statistically strong validation.",
    }
    out = root / "outputs" / "reports" / "radiate_tiny_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
