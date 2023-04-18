from pathlib import Path

import pandas as pd


def gather_metrics(metrics_dir: str) -> None:
    metrics = []
    for model_metrics_path in Path(metrics_dir).glob("*.csv"):
        model_metrics = pd.read_csv(model_metrics_path).set_index("model")
        metrics.append(model_metrics)
    metrics = pd.concat(metrics, axis=0).sort_values(by="map", ascending=False).dropna(axis=1)

    return metrics


if __name__ == "__main__":
    print(gather_metrics("data/metrics/"))
