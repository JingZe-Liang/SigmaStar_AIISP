import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
LABELS = ("a3_ema", "fixed1024_ema", "refresh1024_ema")
MODES = ("static", "global_translation", "moving_patch")
MODE_TITLES = ("Static", "Global translation", "Moving patch")
CASES = [
    f"{scene}__{mode}__gain_{gain}"
    for mode in MODES
    for scene in ("detail", "dark")
    for gain in ("800", "12800")
]


def load_results() -> dict[str, dict]:
    results = {}
    protocol = None
    manifest = None
    for label in LABELS:
        folder = "a3" if label == "a3_ema" else label
        source = ROOT / f"artifacts/a12_paired_eval_{folder}_20261030/probes/evaluation.json"
        document = json.loads(source.read_text())
        if label == "a3_ema":
            protocol = document["metric_protocol"]
            manifest = document["prepared_manifest_sha256"]
        else:
            assert document["metric_protocol"] == protocol
            assert document["prepared_manifest_sha256"] == manifest
        assert document["weights"] == "ema"
        cases = {case["name"]: case for case in document["cases"]}
        assert len(cases) == len(CASES) and set(cases) == set(CASES)
        results[label] = {}
        for name in CASES:
            case = cases[name]
            region = "all" if case["mode"] == "static" else "motion_swept"
            metrics = case[label][region]
            assert np.isfinite(metrics["rmse_dn"])
            assert np.isfinite(metrics["highpass_rmse_dn"])
            results[label][name] = metrics
    for name in CASES:
        for key in ("pixels", "highpass_pixels"):
            assert len({results[label][name][key] for label in LABELS}) == 1
    return results


def main() -> None:
    results = load_results()
    fixed_delta = np.array([
        results["fixed1024_ema"][name]["rmse_dn"] - results["a3_ema"][name]["rmse_dn"]
        for name in CASES
    ])
    refresh_delta = np.array([
        results["refresh1024_ema"][name]["rmse_dn"] - results["a3_ema"][name]["rmse_dn"]
        for name in CASES
    ])
    highpass_delta = np.array([
        results["refresh1024_ema"][name]["highpass_rmse_dn"]
        - results["fixed1024_ema"][name]["highpass_rmse_dn"]
        for name in CASES
    ])
    positions = np.array([group * 5 + row for group in range(3) for row in range(4)])
    tick_labels = [
        f"{name.split('__')[0].capitalize()} / {int(name.rsplit('_', 1)[1]):,}"
        for name in CASES
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "a12-paired-probes-20261030",
    })
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 8.4), sharey=True,
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    figure.subplots_adjust(left=0.21, right=0.97, top=0.82, bottom=0.235, wspace=0.16)
    figure.suptitle("Paired sample-refresh diagnostic", x=0.21, y=0.97,
                   ha="left", fontsize=18, fontweight="bold")
    figure.text(0.21, 0.925,
                "EMA at 1,024 updates; shared 512-update start. Each row is one probe case.",
                fontsize=11, color="#444444")

    for axis in (left, right):
        for group in range(3):
            axis.axhspan(group * 5 - 0.45, group * 5 + 3.45,
                         color="#f4f6f8" if group % 2 == 0 else "#ffffff", zorder=0)
        axis.axvline(0, color="#333333", linewidth=1.1, zorder=2)
        axis.grid(axis="x", color="#dce0e4", linewidth=0.65, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#a9b0b7")
        axis.tick_params(axis="y", length=0, pad=10)
        axis.set_ylim(13.7, -0.65)

    left.scatter(fixed_delta, positions - 0.13, color="#b86b2d", marker="o", s=43,
                 label="Fixed EMA", zorder=4)
    left.scatter(refresh_delta, positions + 0.13, color="#23699d", marker="D", s=32,
                 label="Refresh EMA", zorder=4)
    left.set_yticks(positions, tick_labels)
    left.set_xlim(-2.6, 6.8)
    left.set_xticks([-2, 0, 2, 4, 6])
    left.set_xlabel("Reconstruction RMSE change vs A3 (DN)", labelpad=10)
    left.set_title("A  |  Candidate minus A3", loc="left", pad=20)
    figure.legend(*left.get_legend_handles_labels(), loc="upper left",
                  bbox_to_anchor=(0.205, 0.905), frameon=False, fontsize=10, ncol=2)
    for group, title in enumerate(MODE_TITLES):
        left.text(-0.39, group * 5 - 0.72, title,
                  transform=left.get_yaxis_transform(), ha="left", va="center",
                  fontsize=11, fontweight="bold", clip_on=False)

    colors = ["#ad5936" if value > 0 else "#357d73" for value in highpass_delta]
    right.barh(positions, highpass_delta, height=0.50, color=colors, zorder=3)
    for y, value in zip(positions, highpass_delta, strict=True):
        offset = 0.008 if value >= 0 else -0.008
        right.text(value + offset, y, f"{value:+.3f}",
                   ha="left" if value >= 0 else "right", va="center", fontsize=9.5)
    right.set_xlim(-0.225, 0.225)
    right.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    right.set_xlabel("High-pass RMSE change: refresh minus fixed (DN)", labelpad=10)
    right.set_title("B  |  Refresh minus fixed", loc="left", pad=20)
    right.tick_params(axis="y", labelleft=False)

    figure.text(0.21, 0.105,
                "Negative = lower error. Panels use different horizontal scales. "
                "Row labels: scene / synthetic noise gain.",
                fontsize=10, color="#444444")
    figure.text(0.21, 0.070,
                "Static cases: whole patch. Motion cases: swept region. "
                "Training-reference-derived synthetic diagnostic.",
                fontsize=10, color="#444444")
    figure.text(0.21, 0.035,
                "Single seed; no significance claim. Not independent real-motion or cross-sensor validation.",
                fontsize=10, color="#444444")
    figure.savefig(OUTPUT / "probe_tradeoff.png", dpi=200,
                   metadata={"Description": "A12 paired sample-refresh EMA diagnostic; no averages."})
    figure.savefig(OUTPUT / "probe_tradeoff.svg", metadata={"Date": "2026-10-30"})
    svg_path = OUTPUT / "probe_tradeoff.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(figure)
    print(json.dumps({
        "cases": len(CASES),
        "refresh_rmse_lower_than_fixed": int(np.sum(refresh_delta < fixed_delta)),
        "moving_patch_highpass_higher_than_fixed": int(np.sum(highpass_delta[-4:] > 0)),
        "outputs": [str(OUTPUT / f"probe_tradeoff.{extension}") for extension in ("png", "svg")],
    }, indent=2))


if __name__ == "__main__":
    main()
