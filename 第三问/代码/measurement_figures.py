"""Regenerate the two figures embedded in the question 3 report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from measurement_run import RAD, R, quadrature


QUESTION = Path(__file__).resolve().parents[1]
IMAGES = QUESTION / "图片"
BLUE, ORANGE, TEAL = "#235789", "#AF522B", "#217A70"


def vertical_percent_axis(ax, title):
    ax.set_ylabel("\n".join(title), rotation=0, va="center", ha="center",
                  labelpad=16, linespacing=1.18)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def probability_plot(result):
    critical = result["s_m"]
    distance = np.unique(np.append(np.linspace(0, 80, 161), critical))
    probability = quadrature(8192)
    coverage = [100 * probability(0.3 + 0.02 * s) for s in distance]

    fig, ax = plt.subplots(figsize=(8.0, 4.25), layout="constrained")
    ax.plot(distance, coverage, color=BLUE, linewidth=2.5)
    ax.axhline(95, linestyle="--", linewidth=1.2, color=ORANGE)
    ax.vlines(critical, 90, 95, linestyle=":", linewidth=1.1, color=ORANGE)
    ax.scatter(critical, 95, color=ORANGE, s=44, zorder=3)
    ax.annotate(f"临界里程 {critical:.3f} m", xy=(critical, 95),
                xytext=(-14, -26), textcoords="offset points", ha="right",
                color=ORANGE, arrowprops={"arrowstyle": "-", "color": ORANGE})
    ax.text(1, 95.22, "95% 达标线", color=ORANGE)
    ax.scatter(distance[-1], coverage[-1], color=BLUE, s=30, zorder=3)
    ax.annotate(f"80 m · {coverage[-1]:.2f}%", (80, coverage[-1]),
                xytext=(-9, -12), textcoords="offset points", ha="right", color=BLUE)
    ax.set(xlim=(0, 82), ylim=(90, 100.5), xticks=np.arange(0, 81, 10),
           yticks=[90, 92, 94, 95, 96, 98, 100],
           xlabel="累计行驶里程（m）")
    vertical_percent_axis(ax, "达标概率")
    ax.grid(axis="y", color="#DEE4E8", lw=0.6)
    path = IMAGES / "达标概率随累计里程变化.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def contribution_plot(result):
    critical = result["s_m"]
    distances = np.array([0, 20, 40, critical, 80])
    position = ((0.3 + 0.02 * distances) / R) ** 2
    variance = np.column_stack((position, np.full_like(position, (0.8 * RAD) ** 2),
                                np.full_like(position, (0.6 * RAD) ** 2)))
    shares = 100 * variance / variance.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.45), layout="constrained")
    x = np.arange(len(distances))
    width = 0.245
    for i, (source, color) in enumerate(zip(("车辆定位", "云台零位", "云台控制"),
                                             (BLUE, ORANGE, TEAL))):
        bars = ax.bar(x + (i - 1) * width, shares[:, i], width * 0.93,
                      color=color, label=source)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, ["0", "20", "40", f"{critical:.3f}\n（临界）", "80"])
    ax.set(ylim=(0, 100), xlabel="累计行驶里程（m）")
    vertical_percent_axis(ax, "一阶角方差占比")
    ax.grid(axis="y", color="#DEE4E8", lw=0.6)
    ax.legend(loc="upper left", ncols=3, frameon=False)
    path = IMAGES / "不同里程下三类误差的贡献.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"],
                         "font.size": 10, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})
    data = json.loads((QUESTION / "文档" / "measurement_results.json").read_text(encoding="utf8"))
    path = probability_plot(data["measurement"])
    print(f"已保存：{path}")
    path = contribution_plot(data["measurement"])
    print(f"已保存：{path}")
