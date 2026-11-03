"""Reproduce the report figures from the accompanying source-derived CSVs."""

import csv
import os
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ASSET_DIR.parents[3]
MPL_CACHE = PROJECT_DIR / "artifacts/report_djc_20261027/mplconfig"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


def read_csv(name: str) -> list[dict[str, str]]:
    with (ASSET_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(ASSET_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(ASSET_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training() -> None:
    exposure = read_csv("exposure_validation.csv")
    v3 = read_csv("v3_validation.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")

    for axis, rows in zip(axes, [exposure, v3], strict=True):
        steps = np.asarray([int(row["step"]) for row in rows])
        values = np.asarray([float(row["validation_proxy_mse"]) for row in rows])
        axis.plot(steps, values, color="#28648B", linewidth=1.9, label="Fixed validation")
        axis.axhline(values[0], color="#737373", linestyle="--", linewidth=1.3,
                     label="Initial / frozen baseline")
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Mixed-supervision proxy MSE")
        axis.grid(axis="y", color="#DFE4E8", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        formatter.set_useOffset(False)
        axis.yaxis.set_major_formatter(formatter)

    steps = [int(row["step"]) for row in exposure]
    values = [float(row["validation_proxy_mse"]) for row in exposure]
    axes[0].scatter(steps, values, color="#28648B", s=22, zorder=3)
    axes[0].set_title("(a) ExposureDiffusion backbone adaptation\n40 fixed development crops", loc="left")
    axes[0].set_xlim(-30, 1050)
    reduction = (1 - values[-1] / values[0]) * 100
    axes[0].annotate(f"Step 1000: {reduction:.3f}% lower proxy error",
                     xy=(1000, values[-1]), xytext=(310, values[0] - 0.000013),
                     arrowprops={"arrowstyle": "-", "color": "#28648B"},
                     fontsize=9, color="#204D6A")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    v3_values = {int(row["step"]): float(row["validation_proxy_mse"]) for row in v3}
    assert min(v3_values, key=v3_values.get) == 2300
    baseline = v3_values[0]
    best_reduction = (1 - v3_values[2300] / baseline) * 100
    final_increase = (v3_values[5000] / baseline - 1) * 100
    axes[1].set_title("(b) V3 output residual training\n20 fixed development crops; equal-domain mean", loc="left")
    axes[1].set_xlim(-100, 5300)
    axes[1].axvline(2300, color="#5B937B", linewidth=1, linestyle=":")
    axes[1].scatter([2300], [v3_values[2300]], color="#267357", marker="D", s=40, zorder=4)
    axes[1].scatter([5000], [v3_values[5000]], color="#BC6B2F", marker="s", s=36, zorder=4)
    axes[1].annotate(f"Best: 2300\n{best_reduction:.3f}% lower",
                     xy=(2300, v3_values[2300]), xytext=(2850, v3_values[2300] + 0.0000010),
                     arrowprops={"arrowstyle": "-", "color": "#267357"},
                     fontsize=9, color="#206048")
    axes[1].annotate(f"Final: 5000\n{final_increase:.3f}% higher",
                     xy=(5000, v3_values[5000]), xytext=(3700, v3_values[5000] + 0.0000012),
                     arrowprops={"arrowstyle": "-", "color": "#BC6B2F"},
                     fontsize=9, color="#955224")
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    figure.supxlabel("Separate validation protocols and vertical scales; lower proxy error is better.",
                     fontsize=9, color="#555555")
    save_figure(figure, "djc_training_curves")


def plot_domains() -> None:
    rows = read_csv("v3_domain_comparison.csv")
    best = np.asarray([float(row["best_improvement_percent"]) for row in rows])
    final = np.asarray([float(row["improvement_percent"]) for row in rows])
    for row in rows:
        baseline = float(row["base"])
        assert abs((1 - float(row["best_loss"]) / baseline) * 100
                   - float(row["best_improvement_percent"])) < 1e-10
        assert abs((1 - float(row["trained"]) / baseline) * 100
                   - float(row["improvement_percent"])) < 1e-10

    figure, axis = plt.subplots(figsize=(9.2, 4.0), layout="constrained")
    positions = np.arange(len(rows))
    for offset, values, color, label in [
        (-0.18, best, "#28648B", "Best validation checkpoint: step 2300"),
        (0.18, final, "#C88642", "Training end: step 5000"),
    ]:
        bars = axis.barh(positions + offset, values, height=0.31, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            axis.text(value + (0.12 if value >= 0 else -0.12),
                      bar.get_y() + bar.get_height() / 2, f"{value:+.3f}%",
                      ha="left" if value >= 0 else "right", va="center", fontsize=9)

    labels = [row["sensor"].replace("465_unspecified", "465 (unspecified)") for row in rows]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_xlim(-8.1, 7.4)
    axis.set_xlabel("Proxy error reduction relative to frozen backbone (%)")
    axis.set_title("V3 on development validation domains\nPositive: lower reference error; negative: higher reference error", loc="left")
    axis.grid(axis="x", color="#DFE4E8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(loc="lower left", bbox_to_anchor=(0, -0.31), ncols=2,
                fontsize=8.5, frameon=False)
    save_figure(figure, "djc_domain_results")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_training()
    plot_domains()
