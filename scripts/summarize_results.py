"""Print compact summaries of locked paper result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def show_table(path: str, columns: list[str] | None = None) -> None:
    table = pd.read_csv(RESULTS / path)
    if columns is not None:
        table = table[columns]
    print(f"\n== {path} ==")
    print(table.to_string(index=False))


def main() -> None:
    show_table(
        "canonical_main_overall.csv",
        ["method", "masked_mae_mean", "masked_rmse_mean", "runs"],
    )
    show_table(
        "plugin_baselines_overall.csv",
        ["method", "masked_mae_mean", "masked_rmse_mean", "runs"],
    )
    show_table(
        "component_ablation_overall.csv",
        ["method", "masked_mae_mean", "masked_rmse_mean", "runs"],
    )


if __name__ == "__main__":
    main()
