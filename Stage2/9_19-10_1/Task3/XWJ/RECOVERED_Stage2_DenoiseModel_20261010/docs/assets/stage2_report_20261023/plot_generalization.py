from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Values from the frozen zero-shot audit, 2026-10-12, static disjoint frame pools.
labels = ["SC465SL\n256x", "SC465SL\n2000x", "sc635hai\n131x"]
even_gain = np.array([13.211, 10.523, 13.588])
odd_gain = np.array([13.158, 10.613, 13.587])
even_hf = np.array([0.957, 0.911, 0.919])
odd_hf = np.array([0.958, 0.912, 0.921])
x = np.arange(len(labels))

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.4), layout="constrained")
for offset, gain, hf, color, label in [
    (-0.16, even_gain, even_hf, "#3978a6", "Even input / odd reference"),
    (0.16, odd_gain, odd_hf, "#cf8b3c", "Odd input / even reference"),
]:
    bars = axes[0].bar(x + offset, gain, width=0.30, color=color, label=label)
    axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    axes[1].scatter(x + offset, hf, color=color, s=42)
    for position, value in zip(x + offset, hf, strict=True):
        axes[1].annotate(f"{value:.3f}", (position, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=8)
for ax in axes:
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.20)
axes[0].set_ylim(0, 16.8)
axes[0].set_ylabel("PSNR improvement over input (dB)")
axes[0].set_title("(a) Method 3: zero-shot static denoising")
axes[0].legend(loc="upper center", fontsize=7.5, frameon=False)
axes[1].axhline(1.0, color="#777777", lw=1, ls="--")
axes[1].set_ylim(0.85, 1.035)
axes[1].set_ylabel("Output / reference high-frequency RMS")
axes[1].set_title("(b) Method 3: detail retention")
fig.savefig(Path(__file__).with_name("generalization_static.pdf"))
plt.close(fig)
