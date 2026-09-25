"""ipywidgets front end for the simulator, for testing in Jupyter before deployment.

    from iot_detector.notebook_ui import SimulatorUI
    SimulatorUI().show()
"""

import asyncio
import html

import ipywidgets as w
import numpy as np
from IPython.display import display
from matplotlib.figure import Figure

from .injectors import INJECTORS
from .simulator import Simulator

WINDOW = 150  # readings shown on the charts
TICK_SECONDS = 0.5

TRUTH_COLOR = "#f4a259"
DETECT_COLOR = "#d62828"
LINE_COLOR = "#1d3557"


class SimulatorUI:
    def __init__(self, simulator: Simulator | None = None, seed=None):
        self.sim = simulator or Simulator(seed=seed)
        self._task = None
        self._build_widgets()
        self._redraw()

    # ---------- widgets ----------
    def _build_widgets(self):
        self.play = w.ToggleButton(value=False, description="Play", icon="play", button_style="success")
        self.step_btn = w.Button(description="Step", icon="step-forward")
        self.reset_btn = w.Button(description="Reset", icon="refresh", button_style="warning")
        self.speed = w.IntSlider(value=4, min=1, max=10, description="Readings/s")
        self.mode = w.Dropdown(options=[("Live (you inject)", "live"), ("Replay test set", "replay")], value=self.sim.mode, description="Mode")
        self.climate = w.Dropdown(
            options=[("Recorded room", "recorded"), ("Synthetic (unseen)", "synthetic")],
            value=self.sim.climate,
            description="Climate",
        )

        self.inject_btns = []
        for kind, spec in INJECTORS.items():
            btn = w.Button(description=spec.label, tooltip=spec.description, layout=w.Layout(width="130px"))
            btn.button_style = "danger" if spec.truth_type in ("dos", "mitm") else "info"
            btn.on_click(lambda _, k=kind: self._inject(k))
            self.inject_btns.append(btn)
        self.intensity = w.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.1, description="Intensity")
        self.length = w.IntText(value=0, description="Length", tooltip="Readings the injection lasts; 0 = default")

        det = self.sim.detector
        self.use_supervised = w.Checkbox(value=det.use_supervised, description="Random Forest")
        self.use_novelty = w.Checkbox(value=det.use_novelty, description="Isolation Forest")
        self.use_sensor = w.Checkbox(value=det.use_sensor_checks, description="Sensor checks")
        self.threshold = w.FloatSlider(value=det.threshold, min=0.05, max=0.95, step=0.05, description="RF threshold")

        self.chart = w.Output()
        self.scoreboard = w.HTML()
        self.latest = w.HTML()
        self.events = w.HTML()

        self.play.observe(self._on_play, "value")
        self.step_btn.on_click(lambda _: self._advance(1))
        self.reset_btn.on_click(lambda _: self._reset())
        self.mode.observe(self._on_mode, "value")
        self.climate.observe(self._on_climate, "value")
        for cb in (self.use_supervised, self.use_novelty, self.use_sensor):
            cb.observe(self._on_detector_settings, "value")
        self.threshold.observe(self._on_detector_settings, "value")

        self.layout = w.VBox(
            [
                w.HTML("<b>Simulation</b>"),
                w.HBox([self.play, self.step_btn, self.reset_btn, self.speed]),
                w.HBox([self.mode, self.climate]),
                w.HTML("<b>Inject</b> <span style='opacity:.7'>(red = attacks, blue = sensor faults)</span>"),
                w.HBox(self.inject_btns[:4]),
                w.HBox(self.inject_btns[4:]),
                w.HBox([self.intensity, self.length]),
                w.HTML("<b>Detector layers</b>"),
                w.HBox([self.use_supervised, self.use_novelty, self.use_sensor, self.threshold]),
                self.chart,
                w.HBox([self.scoreboard, self.latest], layout=w.Layout(gap="24px")),
                self.events,
            ]
        )

    def show(self):
        display(self.layout)

    # ---------- actions ----------
    def _inject(self, kind):
        self.sim.inject(kind, self.intensity.value, self.length.value or None)
        if not self.play.value:
            self._advance(1)

    def _advance(self, n):
        for _ in range(n):
            self.sim.step()
        self._redraw()

    def _reset(self):
        self.sim.reset()
        self._redraw()

    def _on_mode(self, change):
        self.sim.mode = change["new"]
        self._reset()

    def _on_climate(self, change):
        self.sim.climate = change["new"]
        self._reset()

    def _on_detector_settings(self, _):
        det = self.sim.detector
        det.use_supervised = self.use_supervised.value
        det.use_novelty = self.use_novelty.value
        det.use_sensor_checks = self.use_sensor.value
        det.threshold = self.threshold.value

    def _on_play(self, change):
        if change["new"]:
            self.play.description, self.play.icon, self.play.button_style = "Pause", "pause", ""
            self._task = asyncio.get_event_loop().create_task(self._loop())
        else:
            self.play.description, self.play.icon, self.play.button_style = "Play", "play", "success"
            if self._task:
                self._task.cancel()

    async def _loop(self):
        # Awaiting between ticks lets the kernel handle button clicks while the stream runs.
        try:
            while self.play.value:
                self._advance(max(1, round(self.speed.value * TICK_SECONDS)))
                await asyncio.sleep(TICK_SECONDS)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # show errors instead of silently stopping
            self.play.value = False
            self.events.value = f"<pre style='color:{DETECT_COLOR}'>{html.escape(repr(exc))}</pre>"

    # ---------- rendering ----------
    def _redraw(self):
        history = self.sim.recent(WINDOW)
        fig = Figure(figsize=(11, 6.5))
        axes = fig.subplots(3, 1, sharex=True)
        series = [("Temperature", "Temperature (°C)", False), ("Humidity", "Humidity (%)", False), ("duration", "Duration (log)", True)]

        if history:
            t = np.array([r["t"] for r in history])
            truth = np.array([r["truth_label"] for r in history])
            flagged = np.array([r["anomaly"] for r in history])
            for ax, (key, label, log) in zip(axes, series):
                y = np.array([r[key] for r in history], dtype=float)
                for start, end in _spans(t, truth):
                    ax.axvspan(start - 0.5, end + 0.5, color=TRUTH_COLOR, alpha=0.25, lw=0)
                ax.plot(t, y, color=LINE_COLOR, lw=1.2)
                ax.scatter(t[flagged], y[flagged], color=DETECT_COLOR, s=16, zorder=3)
                ax.set_ylabel(label)
                if log:
                    ax.set_yscale("log")
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel("Reading")
        else:
            axes[0].set_title("Press Play or Step to start the stream")
        axes[0].legend(
            handles=[
                _patch(TRUTH_COLOR, "True anomaly (injected or labelled)"),
                _dot(DETECT_COLOR, "Flagged by detector"),
            ],
            loc="upper left",
            fontsize=8,
        )
        fig.tight_layout()

        with self.chart:
            self.chart.clear_output(wait=True)
            display(fig)
        self._render_tables()

    def _render_tables(self):
        sb = self.sim.scoreboard()
        pct = lambda v: "–" if v is None else f"{v:.1%}"
        latency = "–" if sb["mean_latency"] is None else f"{sb['mean_latency']:.1f} readings"
        self.scoreboard.value = f"""
        <b>Scoreboard</b>
        <table>
          <tr><td></td><td><b>Flagged</b></td><td><b>Not flagged</b></td></tr>
          <tr><td><b>Anomaly</b></td><td>{sb['tp']} (caught)</td><td>{sb['fn']} (missed)</td></tr>
          <tr><td><b>Normal</b></td><td>{sb['fp']} (false alarm)</td><td>{sb['tn']}</td></tr>
        </table>
        Precision {pct(sb['precision'])} · Recall {pct(sb['recall'])} · False alarms {pct(sb['false_alarm_rate'])}<br>
        Injections caught {sb['events_detected']}/{sb['events']} · Mean time to detect {latency}
        """

        if self.sim.history:
            r = self.sim.history[-1]
            verdict = f"<span style='color:{DETECT_COLOR}'><b>ANOMALY</b></span>" if r["anomaly"] else "<b>normal</b>"
            self.latest.value = f"""
            <b>Latest reading (t={r['t']})</b><br>
            Temp {r['Temperature']} °C · Humidity {r['Humidity']} % · Status {r['Status']}<br>
            Duration {r['duration']} · bytes {r['destination_bytes']}/{r['missed_bytes']}/{r['dst_ip_bytes']}<br>
            Truth: {r['truth_type']} · Detector: {verdict}<br>
            P(anomaly) {r['p_anomaly']:.2f} · type guess {r['attack_type']} · novelty {r['novelty_score']:.3f}<br>
            <i>{html.escape(r['reasons']) or 'no layer fired'}</i>
            """
        else:
            self.latest.value = ""

        events = self.sim.events_frame().tail(8).iloc[::-1]
        if events.empty:
            self.events.value = ""
            return
        rows = "".join(
            f"<tr><td>{e.event_id}</td><td>{html.escape(e.label)}</td><td>{e.intensity:.1f}</td><td>{e.start}</td>"
            f"<td>{e.length}</td><td>{'yes, after ' + str(int(e.latency)) if e.detected else 'no'}</td></tr>"
            for e in events.itertuples()
        )
        self.events.value = (
            "<b>Injections</b><table><tr><th>#</th><th>Type</th><th>Intensity</th><th>Start</th>"
            f"<th>Length</th><th>Detected</th></tr>{rows}</table>"
        )


def _spans(t, mask):
    """Contiguous (start, end) runs of t where mask is 1."""
    spans, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        if not m and start is not None:
            spans.append((start, prev))
            start = None
        prev = ti
    if start is not None:
        spans.append((start, prev))
    return spans


def _patch(color, label):
    from matplotlib.patches import Patch

    return Patch(color=color, alpha=0.4, label=label)


def _dot(color, label):
    from matplotlib.lines import Line2D

    return Line2D([], [], marker="o", color=color, lw=0, label=label)
