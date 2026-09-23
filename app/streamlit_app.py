"""IoT anomaly injector and detector simulator: Streamlit front end.

Run locally:  streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works without `pip install -e .` (e.g. Streamlit Cloud)
    sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from iot_detector import MODEL_DIR
from iot_detector.detector import Detector, StreamDetector
from iot_detector.injectors import INJECTORS
from iot_detector.simulator import Simulator

WINDOW = 200  # readings shown on the charts
TICK_SECONDS = 1.0
MAX_HISTORY = 5000  # readings kept per session (scoreboard still covers the whole run)

TRUTH_COLOR = "#f4a259"
DETECT_COLOR = "#e63946"
LINE_COLOR = "#4c78a8"
SERIES = [
    ("Temperature", "Temperature (°C)", "linear"),
    ("Humidity", "Humidity (%)", "linear"),
    ("duration", "Connection duration", "log"),
]
MODE_LABELS = {"live": "Live: you inject anomalies", "replay": "Replay: real test set"}
CLIMATE_LABELS = {"recorded": "Recorded room", "synthetic": "Synthetic (unseen by model)"}

st.set_page_config(page_title="IoT Anomaly Simulator", page_icon="📡", layout="wide")


# ---------- shared resources (loaded once per server) ----------
@st.cache_resource
def load_detector() -> Detector:
    return Detector(MODEL_DIR)


@st.cache_resource
def load_reference() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "reference.csv")


if not (MODEL_DIR / "metadata.json").exists():
    st.error("No trained models found. Run `python -m iot_detector.train` first.")
    st.stop()


# ---------- per-session state ----------
def get_sim() -> Simulator:
    if "sim" not in st.session_state:
        st.session_state.sim = Simulator(
            detector=StreamDetector(load_detector()),
            reference=load_reference(),
            max_history=MAX_HISTORY,
        )
        st.session_state.last_tick = 0.0
    return st.session_state.sim


sim = get_sim()


def inject(kind: str):
    sim.inject(kind, st.session_state.intensity, st.session_state.length or None)
    if not st.session_state.running:
        sim.step()  # show the effect straight away when paused


def apply_source_settings():
    sim.mode = st.session_state.mode
    sim.climate = st.session_state.climate
    sim.reset()


def step_once():
    sim.step()


# ---------- sidebar: simulation and detector settings ----------
with st.sidebar:
    st.header("Simulation")
    st.toggle("Run stream", key="running")
    st.slider("Readings per second", 1, 10, 4, key="speed")
    c1, c2 = st.columns(2)
    c1.button("Step", icon=":material/skip_next:", on_click=step_once, width="stretch", disabled=st.session_state.running)
    c2.button("Reset", icon=":material/restart_alt:", on_click=sim.reset, width="stretch")
    st.radio("Data source", list(MODE_LABELS), format_func=MODE_LABELS.get, key="mode", on_change=apply_source_settings)
    st.radio(
        "Climate",
        list(CLIMATE_LABELS),
        format_func=CLIMATE_LABELS.get,
        key="climate",
        on_change=apply_source_settings,
        disabled=st.session_state.mode == "replay",
        help="Synthetic moves temperature and humidity through ranges the model never saw for normal traffic.",
    )

    st.header("Detector layers")
    det = sim.detector
    det.use_supervised = st.checkbox("Random Forest (known attacks)", value=True)
    det.use_novelty = st.checkbox("Isolation Forest (unlike normal)", value=True)
    det.use_sensor_checks = st.checkbox("Sensor checks (range, jump, flatline)", value=True)
    det.threshold = st.slider("Random Forest threshold", 0.05, 0.95, det.detector.default_threshold, 0.05)


# ---------- header and injection controls ----------
st.title("IoT anomaly simulator")
st.caption(
    "A device stream is scored reading by reading. Inject attacks or sensor faults and watch which "
    "detector layers catch them. Orange bands are true anomalies; red dots are what the detector flagged."
)

with st.container(border=True):
    st.markdown("**Inject an anomaly**")
    s1, s2 = st.columns([3, 1])
    s1.slider("Intensity", 0.0, 1.0, 1.0, 0.1, key="intensity")
    s2.number_input("Length (0 = default)", min_value=0, max_value=500, value=0, key="length")
    attacks = [k for k, s in INJECTORS.items() if s.truth_type != "fault"]
    faults = [k for k, s in INJECTORS.items() if s.truth_type == "fault"]
    for kinds, button_type in ((attacks, "primary"), (faults, "secondary")):
        cols = st.columns(len(attacks))
        for col, kind in zip(cols, kinds):
            spec = INJECTORS[kind]
            col.button(spec.label, key=f"inject_{kind}", help=spec.description, type=button_type,
                       on_click=inject, args=(kind,), width="stretch")


# ---------- rendering ----------
def truth_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of true anomalies as (start, end) for shaded bands."""
    run_id = (df["truth_label"] != df["truth_label"].shift()).cumsum()
    runs = df[df["truth_label"] == 1].groupby(run_id)["t"].agg(["min", "max"])
    return pd.DataFrame({"start": runs["min"] - 0.5, "end": runs["max"] + 0.5})


def stream_chart(df: pd.DataFrame) -> alt.VConcatChart:
    domain = [df["t"].min() - 0.5, df["t"].max() + 0.5]
    x = alt.X("t:Q", title="Reading", scale=alt.Scale(domain=domain, nice=False))
    spans = truth_spans(df)
    flagged = df[df["anomaly"]]
    tooltip = ["t:Q", "Temperature:Q", "Humidity:Q", "duration:Q", "truth_type:N", "p_anomaly:Q", "reasons:N"]

    panels = []
    for i, (key, title, scale) in enumerate(SERIES):
        y_axis = alt.Axis(format="~s", values=[100, 1_000, 10_000, 100_000]) if scale == "log" else alt.Axis()
        y = alt.Y(f"{key}:Q", title=title, scale=alt.Scale(type=scale, zero=False), axis=y_axis)
        px = x if i == len(SERIES) - 1 else x.axis(labels=False, title=None)
        bands = alt.Chart(spans).mark_rect(color=TRUTH_COLOR, opacity=0.3).encode(x=alt.X("start:Q", scale=alt.Scale(domain=domain)), x2="end:Q")
        line = alt.Chart(df).mark_line(color=LINE_COLOR, strokeWidth=1.5).encode(x=px, y=y)
        dots = alt.Chart(flagged).mark_circle(color=DETECT_COLOR, size=45, opacity=1).encode(x=px, y=y, tooltip=tooltip)
        panels.append(alt.layer(bands, line, dots).properties(height=150))
    return alt.vconcat(*panels, spacing=6).resolve_scale(x="shared")


def pct(v):
    return "–" if v is None else f"{v:.1%}"


def render():
    sb = sim.scoreboard()
    m = st.columns(5)
    m[0].metric("Readings", f"{sb['readings']:,}")
    m[1].metric("Recall", pct(sb["recall"]), help="Share of anomalous readings the detector flagged")
    m[2].metric("Precision", pct(sb["precision"]), help="Share of flagged readings that really were anomalous")
    m[3].metric("False alarm rate", pct(sb["false_alarm_rate"]), help="Share of normal readings wrongly flagged")
    m[4].metric(
        "Injections caught",
        f"{sb['events_detected']}/{sb['events']}",
        help="Mean readings until first detection: "
        + ("–" if sb["mean_latency"] is None else f"{sb['mean_latency']:.1f}"),
    )

    recent = sim.recent(WINDOW)
    if not recent:
        st.info("Turn on **Run stream** in the sidebar, or press **Step**, to start.")
        return
    st.altair_chart(stream_chart(pd.DataFrame(recent)), width="stretch")

    left, right = st.columns([2, 3])
    with left:
        r = recent[-1]
        with st.container(border=True):
            st.markdown(f"**Latest reading** · t={r['t']}")
            if r["anomaly"]:
                st.error(f"Anomaly: {r['reasons']}", icon=":material/warning:")
            else:
                st.success("Normal: no layer fired", icon=":material/check_circle:")
            st.markdown(
                f"Temperature **{r['Temperature']} °C** · Humidity **{r['Humidity']} %** · Status **{r['Status']}**  \n"
                f"Duration **{r['duration']}** · bytes {r['destination_bytes']} / {r['missed_bytes']} / {r['dst_ip_bytes']}  \n"
                f"Truth: **{r['truth_type']}** · P(anomaly) {r['p_anomaly']:.2f} · type guess {r['attack_type']} · "
                f"novelty {r['novelty_score']:.3f}"
            )
    with right:
        st.markdown("**Injections**")
        events = sim.events_frame()
        if events.empty:
            st.caption("None yet. Use the buttons above.")
        else:
            events["detected"] = events.apply(
                lambda e: f"after {int(e['latency'])} readings" if e["detected"] else ("running" if e["start"] + e["length"] > sim.t else "missed"),
                axis=1,
            )
            st.dataframe(
                events.iloc[::-1][["event_id", "label", "intensity", "start", "length", "detected"]],
                hide_index=True,
                height=200,
                column_config={"event_id": "#", "label": "Type"},
            )

    with st.expander("Raw readings"):
        history = sim.history_frame()
        st.dataframe(history.iloc[::-1].head(500), hide_index=True, height=300)
        st.download_button("Download CSV", history.to_csv(index=False), "iot_simulation.csv", "text/csv")


@st.fragment(run_every=TICK_SECONDS if st.session_state.running else None)
def live_view():
    # Only the fragment reruns each tick. Full reruns (button clicks) also pass through here,
    # so step only when a tick has actually elapsed.
    now = time.monotonic()
    if st.session_state.running and now - st.session_state.last_tick >= TICK_SECONDS * 0.8:
        for _ in range(int(st.session_state.speed * TICK_SECONDS)):
            sim.step()
        st.session_state.last_tick = now
    render()


live_view()


with st.expander("About the models"):
    meta = load_detector().metadata
    metrics = meta["metrics"]
    st.markdown(
        f"""
Trained on {meta['rows']['train']:,} readings, tested on {meta['rows']['test']:,} held-out readings ({meta['trained_at']}).

| Layer | What it catches | Test accuracy |
|---|---|---|
| Random Forest, normal vs anomaly | DoS and MITM patterns from the dataset | {metrics['binary']['report']['accuracy']:.1%} |
| Random Forest, attack type | Normal, DoS or MITM | {metrics['attack_type']['report']['accuracy']:.1%} |
| Isolation Forest | Readings unlike normal traffic (trained on normal only) | {metrics['novelty']['report']['accuracy']:.1%} |
| Sensor checks | Out-of-range, sudden jumps, frozen sensors | rule-based |

**Known limitation:** the dataset's attacks were all recorded in a narrow temperature and humidity band, so the
models partly rely on climate. Switch climate to *Synthetic* to see false alarms rise with no attack at all.

Columns not used as inputs: {", ".join(f"`{c}`" for c in meta["dropped_columns"])}.
"""
    )
