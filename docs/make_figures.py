"""Regenerate the README figures from the data and the trained models.

Usage:  python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from iot_detector import DATA_PATH
from iot_detector.features import FEATURES, load_dataset
from iot_detector.injectors import INJECTORS
from iot_detector.simulator import Simulator

OUT = Path(__file__).resolve().parent / "images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_BAR = "#c3c2b7"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRAY_BAR,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def hbar(ax, labels, values, colors, fmt):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v + 1, yi, fmt(v), va="center", color=INK, fontsize=10)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_comparison():
    df = load_dataset(DATA_PATH)

    def score(columns):
        X = pd.get_dummies(df[columns], drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, df["Label"], test_size=0.3, random_state=42, stratify=df["Type"])
        return 100 * RandomForestClassifier(random_state=42, class_weight="balanced").fit(X_tr, y_tr).score(X_te, y_te)

    rows = [
        ("Original notebook\n(answer column included)", score(FEATURES + ["Type", "source_ip", "source_port"]), GRAY_BAR),
        ("Given ONLY the answer column\n(no sensor data at all)", score(["Type"]), GRAY_BAR),
        ("Given ONLY the device's\nnetwork address", score(["source_ip"]), GRAY_BAR),
        ("Corrected model\n(sensor and traffic data only)", score(FEATURES), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    hbar(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], lambda v: f"{v:.1f}%")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Accuracy on unseen test data")
    ax.set_title("The 100% score came from shortcuts, not from learning")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_comparison.png", dpi=150)


def detection_by_anomaly():
    order = ["replay_dos", "replay_mitm", "dos", "mitm", "spike", "drift", "stuck"]
    names = {
        "replay_dos": "Real DoS attack",
        "replay_mitm": "Real MITM attack",
        "dos": "Simulated DoS attack",
        "mitm": "Simulated MITM attack",
        "spike": "Sensor spike",
        "drift": "Sensor drift",
        "stuck": "Sensor stuck",
    }
    labels, flagged, colors = [], [], []
    for kind in order:
        s = Simulator(seed=1)
        s.run(30)
        s.inject(kind, 1.0)
        h = s.run(INJECTORS[kind].default_length)
        inj = h[h["injected"].notna()]
        latency = s.events_frame()["latency"].iloc[0]
        when = "caught instantly" if latency == 0 else f"caught after {int(latency)} readings"
        labels.append(f"{names[kind]}\n{when}")
        flagged.append(100 * inj["anomaly"].mean())
        colors.append(BLUE if INJECTORS[kind].truth_type != "fault" else ORANGE)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    hbar(ax, labels, flagged, colors, lambda v: f"{v:.0f}%")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of affected readings flagged (full intensity)")
    ax.set_title("Attacks are caught at once; slow sensor faults take longer")
    ax.legend(handles=[Patch(color=BLUE, label="Cyber-attack"), Patch(color=ORANGE, label="Sensor fault")],
              loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "detection_by_anomaly.png", dpi=150)


def climate_scatter():
    df = load_dataset(DATA_PATH)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = [("normal", BLUE, "Normal", 10, 0.35), ("mitm", AQUA, "MITM attack", 14, 0.7), ("dos", ORANGE, "DoS attack", 14, 0.8)]
    for t, color, label, size, alpha in styles:
        d = df[df["Type"] == t]
        ax.scatter(d["Temperature"], d["Humidity"], s=size, color=color, alpha=alpha, label=label, edgecolors="none")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Attacks were recorded in a cooler, narrower climate than normal traffic")
    ax.legend(frameon=False, loc="upper right", markerscale=2)
    ax.annotate("attacks cluster here", xy=(27.2, 45.5), xytext=(31.5, 38), color=INK_2,
                arrowprops={"arrowstyle": "->", "color": INK_2})
    fig.tight_layout()
    fig.savefig(OUT / "climate_bias.png", dpi=150)


def simulator_timeline():
    s = Simulator(seed=0)
    s.run(30)
    s.inject("dos")
    s.run(30)
    s.inject("spike")
    s.run(15)
    s.inject("drift")
    s.run(70)
    h = s.history_frame()

    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
    series = [("Temperature", "Temperature\n(°C)", False), ("Humidity", "Humidity\n(%)", False), ("duration", "Connection\nduration", True)]
    truth = h["truth_label"].to_numpy()
    runs = (h["truth_label"] != h["truth_label"].shift()).cumsum()
    spans = h[truth == 1].groupby(runs)["t"].agg(["min", "max"])
    for ax, (key, label, log) in zip(axes, series):
        for lo, hi in spans.itertuples(index=False):
            ax.axvspan(lo - 0.5, hi + 0.5, color=ORANGE, alpha=0.16, lw=0)
        ax.plot(h["t"], h[key], color=BLUE, lw=1.6)
        f = h[h["anomaly"]]
        ax.scatter(f["t"], f[key], color=CRITICAL, s=16, zorder=3, edgecolors=SURFACE, linewidths=0.6)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        if log:
            ax.set_yscale("log")
    axes[-1].set_xlabel("Reading number")
    for x, text in [(40, "DoS attack"), (60, "spike"), (110, "sensor drift")]:
        axes[0].text(x, axes[0].get_ylim()[1], text, ha="center", va="bottom", color=INK_2, fontsize=9)
    axes[0].set_title("The simulator: injected problems (shaded) and what the detector flagged (red)", pad=18)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.6, label="Device reading"),
            Patch(color=ORANGE, alpha=0.3, label="Injected problem"),
            Line2D([], [], marker="o", color=CRITICAL, lw=0, label="Flagged by detector"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "simulator_timeline.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_comparison()
    detection_by_anomaly()
    climate_scatter()
    simulator_timeline()
    print(f"Saved figures to {OUT}")
