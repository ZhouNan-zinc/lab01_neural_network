"""手写数字 MLP 对照实验。

运行：python mlp_digits.py --exp all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden: Tuple[int, ...] = (32,)
    act: str = "sigmoid"
    optimizer: str = "sgd"
    lr: float = 0.1
    weight_decay: float = 0.0
    dropout: float = 0.0
    use_bn: bool = False
    epochs: int = 30
    seed: int = 42
    output_dir: str = "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(split_seed: int = 42, device: torch.device | None = None):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=split_seed
    )
    tensors = (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long),
    )
    assert tensors[0].shape == (1437, 64) and tensors[2].shape == (360, 64)
    assert tensors[1].dtype == torch.long and tensors[3].dtype == torch.long
    assert 0 <= float(tensors[0].min()) <= float(tensors[0].max()) <= 1
    if device is not None:
        tensors = tuple(t.to(device) for t in tensors)
    return tensors


class MLP(nn.Module):
    def __init__(self, hidden: Tuple[int, ...] = (32,), act: str = "sigmoid", dropout: float = 0.0, use_bn: bool = False):
        super().__init__()
        if act not in {"sigmoid", "tanh", "relu", "gelu"}:
            raise ValueError(f"unsupported activation: {act}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        act_cls = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        layers: list[nn.Module] = []
        prev = 64
        for width in hidden:
            if width <= 0:
                raise ValueError("hidden widths must be positive")
            layers.append(nn.Linear(prev, width))
            if use_bn:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")


def save_plot(history: Dict[str, list], path: Path, title: str) -> None:
    # Prefer a Chinese font when available, with a safe fallback on Linux.
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, np.asarray(history["acc"]) * 100, label="test accuracy", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Test accuracy")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_one(config: ExperimentConfig, data=None) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data is None:
        data = prepare_data(device=device)
    X_tr, y_tr, X_te, y_te = data
    model = MLP(config.hidden, config.act, config.dropout, config.use_bn).to(device)
    optimizer = make_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = {"loss": [], "acc": []}
    start = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(dim=1) == y_te).float().mean().item()
        history["loss"].append(float(loss.detach().cpu()))
        history["acc"].append(float(acc))
    elapsed = time.perf_counter() - start
    finite = all(math.isfinite(v) for v in history["loss"] + history["acc"])
    result = {
        "name": config.name,
        "config": asdict(config),
        "test_accuracy": history["acc"][-1],
        "elapsed_s": elapsed,
        "finite": finite,
        "history": history,
        "device": str(device),
    }
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"history_{config.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(history, out / config_plot_name(config.name), f"{config.name} MLP")
    print(f"{config.name:>12s} | acc={result['test_accuracy']*100:6.2f}% | time={elapsed:6.3f}s | finite={finite}")
    return result


def config_plot_name(name: str) -> str:
    if name == "baseline":
        return "result_baseline.png"
    if name == "lr_too_big":
        return "result_lr_too_big.png"
    if name.startswith("exp") and name[3:].isdigit():
        return f"result_{name}.png"
    if name.startswith("exp8_run"):
        return f"result_{name}.png"
    return f"result_{name}.png"


def write_metrics(results: Iterable[dict], path: Path) -> None:
    rows = []
    for r in results:
        c = r["config"]
        rows.append({
            "name": r["name"], "hidden": str(c["hidden"]), "act": c["act"], "optimizer": c["optimizer"],
            "lr": c["lr"], "weight_decay": c["weight_decay"], "dropout": c["dropout"], "use_bn": c["use_bn"],
            "epochs": c["epochs"], "seed": c["seed"], "test_accuracy_percent": r["test_accuracy"] * 100,
            "elapsed_s": r["elapsed_s"], "finite": r["finite"], "device": r["device"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def baseline_configs(output_dir: str = "results") -> list[ExperimentConfig]:
    base = ExperimentConfig("baseline", output_dir=output_dir)
    return [
        base,
        replace(base, name="exp1", act="relu"),
        replace(base, name="exp2", hidden=(128, 128)),
        replace(base, name="exp3", optimizer="adam"),
        replace(base, name="exp4", lr=0.01),
        replace(base, name="exp5", weight_decay=1e-4),
        replace(base, name="exp6", dropout=0.2),
        replace(base, name="exp7", use_bn=True),
    ]


def run_all(output_dir: str) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(device=device)
    results = [run_one(c, data) for c in baseline_configs(output_dir)]
    # A validated, practical combination: ReLU + wider/deeper network + BN + Adam.
    best_cfg = ExperimentConfig("exp8", hidden=(128, 128), act="relu", optimizer="adam", lr=0.001, use_bn=True, output_dir=output_dir)
    results.append(run_one(best_cfg, data))
    results.append(run_one(replace(ExperimentConfig("lr_too_big", output_dir=output_dir), lr=10.0), data))
    write_metrics(results, Path(output_dir) / "metrics.csv")
    return results


def run_best_repeats(output_dir: str, repeats: int = 3) -> dict:
    cfg = ExperimentConfig("exp8", hidden=(128, 128), act="relu", optimizer="adam", lr=0.001, use_bn=True, output_dir=output_dir)
    data = prepare_data(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    runs = []
    for i in range(repeats):
        r = run_one(replace(cfg, name=f"exp8_run{i+1}", seed=42 + i), data)
        runs.append({"run": i + 1, "seed": 42 + i, "test_accuracy_percent": r["test_accuracy"] * 100, "elapsed_s": r["elapsed_s"]})
    acc = np.array([r["test_accuracy_percent"] for r in runs])
    summary = {"config": asdict(cfg), "runs": runs, "mean_accuracy_percent": float(acc.mean()), "std_accuracy_percent": float(acc.std(ddof=1)) if repeats > 1 else 0.0, "min_accuracy_percent": float(acc.min()), "max_accuracy_percent": float(acc.max())}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "best_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["baseline", "all", "best"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.exp == "all":
        run_all(args.output_dir)
    elif args.exp == "baseline":
        run_one(ExperimentConfig("baseline", output_dir=args.output_dir), prepare_data(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    else:
        print(json.dumps(run_best_repeats(args.output_dir, args.repeats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
