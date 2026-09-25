"""IoT anomaly detector: Streamlit front end.

Three pages:
  Check a reading   type in one device reading and see what the AI says about it.
  Watch a stream    readings arrive one by one; the AI judges each and is scored against the real answer.
  How it works      plain-language explanation of the data, the checks and the results.

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
from iot_detector.features import FEATURES
from iot_detector.injectors import INJECTORS
from iot_detector.simulator import Simulator

WINDOW = 200  # readings shown on the charts
TABLE_ROWS = 12  # readings shown in the "latest readings" table
TICK_SECONDS = 1.0
MAX_HISTORY = 5000  # readings kept per session (scoreboard still covers the whole run)

TRUTH_COLOR = "#f4a259"
DETECT_COLOR = "#e63946"
LINE_COLOR = "#4c78a8"
CURSOR_COLOR = "#888888"
SERIES = [
    ("Temperature", "Temperature (°C)", "linear"),
    ("Humidity", "Humidity (%)", "linear"),
    ("duration", "Connection time", "log"),
]

PAGES = {
    "check": ":material/search: Check a reading",
    "stream": ":material/sensors: Watch a live stream",
    "about": ":material/menu_book: How it works",
}
MODE_LABELS = {
    "replay": "Real test data",
    "live": "Simulated device (you add problems)",
}
MODE_HELP = {
    "replay": (
        "**What's happening:** we play back **{n:,} real readings** from the dataset, one at a time, in the order "
        "they were recorded. The AI **never saw these readings while it was learning**. Each one already has a "
        "known answer (normal, DoS attack or MITM attack), so for every reading we compare what the AI said with "
        "the real answer and keep score."
    ),
    "live": (
        "**What's happening:** a healthy device sends normal readings (real normal readings from the test data). "
        "Nothing is wrong until **you press one of the problem buttons below**. That changes the next readings to "
        "look like an attack or a broken sensor. Watch whether the AI notices, and how quickly."
    ),
}
CLIMATE_LABELS = {"recorded": "Real room climate", "synthetic": "Made-up climate (unseen by the AI)"}
TYPE_NAMES = {"normal": "Normal", "dos": "DoS attack", "mitm": "MITM attack", "fault": "Sensor fault"}

# Plain-language description of each input field, shown on the "Check a reading" page.
FIELDS = {
    "Temperature": ("Temperature (°C)", "Room temperature the sensor reported."),
    "Humidity": ("Humidity (%)", "Air humidity the sensor reported."),
    "duration": ("Connection time", "How long the device's network connection lasted (dataset units)."),
    "destination_bytes": ("Data sent to device (bytes)", "Amount of data sent to the device."),
    "missed_bytes": ("Lost data (bytes)", "Data that went missing on the way and had to be resent."),
    "dst_ip_bytes": ("Total traffic incl. headers (bytes)", "All data sent to the device, including network overhead."),
}
FIELD_LIMITS = {  # (min, max, step) for the number inputs
    "Temperature": (-10.0, 60.0, 0.1),
    "Humidity": (0.0, 100.0, 0.1),
    "duration": (0, 300_000, 1),
    "destination_bytes": (0, 10_000, 1),
    "missed_bytes": (0, 10_000, 1),
    "dst_ip_bytes": (0, 10_000, 1),
}

st.set_page_config(
    page_title="IoT Anomaly Detector", page_icon="📡", layout="wide", initial_sidebar_state="collapsed"
)


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
            mode="replay",
            max_history=MAX_HISTORY,
        )
        st.session_state.last_tick = 0.0
        st.session_state.running = False
        st.session_state.cursor = None  # reading being reviewed; None = follow the newest
    return st.session_state.sim


sim = get_sim()
det = sim.detector
test_rows = load_reference().query("split == 'test'")


# ---------- plain-language helpers ----------
def explain_reason(reason: str) -> str:
    if reason.startswith("model: "):
        kind = reason.removeprefix("model: ")
        return "looks like an attack" if kind == "anomaly" else f"looks like a {TYPE_NAMES.get(kind, kind)}"
    if reason == "unlike normal traffic":
        return "traffic is unusual"
    if reason == "sensors flatlined":
        return "sensors stopped changing"
    for sensor in ("Temperature", "Humidity"):
        if reason == f"{sensor} out of range":
            return f"{sensor.lower()} outside normal range"
        if reason == f"{sensor} jump":
            return f"{sensor.lower()} changed too fast"
    return reason


def explain_reasons(reasons: str) -> str:
    return "; ".join(explain_reason(r) for r in reasons.split("; ")) if reasons else ""


def outcome(truth_label: int, flagged: bool) -> str:
    if truth_label and flagged:
        return "✅ Caught"
    if not truth_label and not flagged:
        return "✅ Correct"
    return "❌ Missed" if truth_label else "⚠️ False alarm"


def pct(v):
    return "–" if v is None else f"{v:.0%}"


# ---------- sidebar: advanced settings (apply to both pages) ----------
with st.sidebar:
    st.header("Advanced settings")
    st.caption("You don't need to change these. They control which checks the AI runs and how cautious it is.")
    det.use_supervised = st.checkbox(
        "Attack recogniser", value=True, key="use_supervised",
        help="Random Forest. Learned what DoS and MITM attacks look like from past examples.",
    )
    det.use_novelty = st.checkbox(
        "Unusual-traffic detector", value=True, key="use_novelty",
        help="Isolation Forest. Learned only what normal looks like and flags anything that doesn't fit.",
    )
    det.use_sensor_checks = st.checkbox(
        "Sensor sanity checks", value=True, key="use_sensor_checks",
        help="Simple rules: value out of range, changed too fast, or stopped changing.",
    )
    det.threshold = st.slider(
        "Alarm threshold", 0.05, 0.95, det.detector.default_threshold, 0.05, key="threshold",
        help="The attack recogniser raises an alarm when its attack likelihood is at or above this. "
        "Lower = more alarms (catches more, more false alarms). Higher = fewer alarms.",
    )


# ---------- header and navigation ----------
st.title("📡 IoT Anomaly Detector")
st.markdown(
    "An AI that watches a smart-home device (a temperature and humidity sensor) and raises an alarm when "
    "something looks wrong: a **cyber-attack** or a **faulty sensor**."
)
page = st.radio("Page", list(PAGES), format_func=PAGES.get, horizontal=True, key="page", label_visibility="collapsed")
st.divider()


# =====================================================================
# Page 1: check a single reading
# =====================================================================
def load_example(kind: str):
    row = test_rows[test_rows["Type"] == kind].sample(1).iloc[0]
    values = {}
    for f in FEATURES:
        v = row[f]
        values[f] = str(v) if f == "Status" else (round(float(v), 1) if f in ("Temperature", "Humidity") else int(v))
        st.session_state[f"in_{f}"] = values[f]
    st.session_state.example = {"kind": kind, "values": values}


def typical_range(field: str) -> str:
    normal = test_rows.loc[test_rows["Type"] == "normal", field]
    lo, hi = normal.quantile(0.05), normal.quantile(0.95)
    fmt = "{:,.1f}" if field in ("Temperature", "Humidity") else "{:,.0f}"
    return f"Normal devices: usually {fmt.format(lo)} – {fmt.format(hi)}"


def check_page():
    if "in_Temperature" not in st.session_state:
        load_example("normal")

    st.subheader("Check a single reading")
    st.markdown(
        "Type in what a device reported and the AI will tell you straight away whether it looks **normal** or "
        "**suspicious**. Change any value and the answer updates. Not sure what to type? Load a real example "
        "from the test data (readings the AI never saw while learning)."
    )
    b = st.columns(3)
    b[0].button("Load a normal example", key="example_normal", on_click=load_example, args=("normal",), width="stretch", icon="🟢")
    b[1].button("Load a DoS attack example", key="example_dos", on_click=load_example, args=("dos",), width="stretch", icon="🔴")
    b[2].button("Load a MITM attack example", key="example_mitm", on_click=load_example, args=("mitm",), width="stretch", icon="🔴")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**🌡️ Sensor readings**")
            c = st.columns(3)
            for col, f in zip(c, ("Temperature", "Humidity")):
                label, help_ = FIELDS[f]
                lo, hi, step = FIELD_LIMITS[f]
                col.number_input(label, lo, hi, step=step, format="%.1f", key=f"in_{f}", help=help_)
                col.caption(typical_range(f))
            c[2].radio("Device switched on?", ["ON", "OFF"], key="in_Status", horizontal=True,
                       help="Whether the device reported itself as on or off.")
        with st.container(border=True):
            st.markdown("**🌐 Network traffic**")
            c = st.columns(2)
            for i, f in enumerate(("duration", "destination_bytes", "missed_bytes", "dst_ip_bytes")):
                label, help_ = FIELDS[f]
                lo, hi, step = FIELD_LIMITS[f]
                col = c[i % 2]
                col.number_input(label, lo, hi, step=step, key=f"in_{f}", help=help_)
                col.caption(typical_range(f))

    reading = {f: st.session_state[f"in_{f}"] for f in FEATURES}
    # A fresh detector per check: this is a single reading, so only the range check of the
    # sensor checks can fire (jumps and frozen sensors need a history).
    checker = StreamDetector(
        load_detector(), threshold=det.threshold, use_supervised=det.use_supervised,
        use_novelty=det.use_novelty, use_sensor_checks=det.use_sensor_checks,
    )
    result = checker.score(reading)
    st.session_state.check_result = result

    with right:
        with st.container(border=True):
            st.markdown("**The AI's verdict**")
            if result["anomaly"]:
                st.error("### 🚨 Suspicious reading")
                st.markdown(f"Why: **{explain_reasons(result['reasons'])}**.")
            else:
                st.success("### ✅ Looks normal")
                st.markdown("None of the checks found anything wrong.")

            example = st.session_state.get("example")
            if example and example["values"] == reading:
                truth = TYPE_NAMES[example["kind"]]
                right_call = (example["kind"] != "normal") == result["anomaly"]
                st.info(
                    f"**Real answer for this example: {truth}.** "
                    + ("The AI got it right. ✅" if right_call else "The AI got this one wrong. ❌")
                )
            elif example:
                st.caption("You've edited the example, so there's no real answer to compare with.")

        with st.container(border=True):
            st.markdown("**What each check said**")
            p = result["p_anomaly"]
            guess = TYPE_NAMES[result["attack_type"]]
            st.markdown("**1. Attack recogniser**" + ("" if det.use_supervised else " _(switched off)_"))
            st.progress(p, text=f"Attack likelihood: {p:.0%} (alarm at {det.threshold:.0%} or more)")
            st.caption(f"Best guess of what this is: **{guess}**")

            st.markdown("**2. Unusual-traffic detector**" + ("" if det.use_novelty else " _(switched off)_"))
            st.markdown("🚩 Doesn't look like normal traffic" if result["novelty_flag"] else "👍 Looks like normal traffic")

            st.markdown("**3. Sensor sanity check**" + ("" if det.use_sensor_checks else " _(switched off)_"))
            limits = load_detector().metadata["sensor_limits"]
            out = [s for s in ("Temperature", "Humidity") if not limits[s]["min"] <= reading[s] <= limits[s]["max"]]
            if out:
                st.markdown("🚩 " + ", ".join(
                    f"{s.lower()} outside the possible range ({limits[s]['min']:.1f} – {limits[s]['max']:.1f})" for s in out
                ))
            else:
                st.markdown("👍 Temperature and humidity are within the normal range")
            st.caption("Checks for sudden jumps or frozen sensors need a series of readings: see *Watch a live stream*.")


# =====================================================================
# Page 2: watch a stream
# =====================================================================
def oldest_t() -> int | None:
    return sim.history[0]["t"] if sim.history else None


def newest_t() -> int | None:
    return sim.history[-1]["t"] if sim.history else None


def go_previous():
    current = newest_t() if st.session_state.cursor is None else st.session_state.cursor
    st.session_state.cursor = max(oldest_t(), current - 1)


def go_next():
    """Move forward through readings already seen; at the newest one, fetch a new reading."""
    cursor = st.session_state.cursor
    if cursor is not None and cursor + 1 < newest_t():
        st.session_state.cursor = cursor + 1
        return
    st.session_state.cursor = None
    if cursor is None:
        sim.step()


def restart():
    sim.reset()
    st.session_state.cursor = None


def inject(kind: str):
    st.session_state.cursor = None
    sim.inject(kind, st.session_state.intensity, st.session_state.length or None)
    if not st.session_state.running:
        sim.step()  # show the effect straight away when paused


def apply_source_settings():
    sim.mode = st.session_state.mode
    sim.climate = st.session_state.get("climate", "recorded")
    restart()


def toggle_running():
    st.session_state.running = not st.session_state.running
    st.session_state.cursor = None


def truth_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of true anomalies as (start, end) for shaded bands."""
    run_id = (df["truth_label"] != df["truth_label"].shift()).cumsum()
    runs = df[df["truth_label"] == 1].groupby(run_id)["t"].agg(["min", "max"])
    return pd.DataFrame({"start": runs["min"] - 0.5, "end": runs["max"] + 0.5})


def stream_chart(df: pd.DataFrame, cursor: int | None) -> alt.VConcatChart:
    domain = [df["t"].min() - 0.5, df["t"].max() + 0.5]
    x = alt.X("t:Q", title="Reading number", scale=alt.Scale(domain=domain, nice=False))
    spans = truth_spans(df)
    flagged = df[df["anomaly"]].assign(
        real_answer=lambda d: d["truth_type"].map(TYPE_NAMES), why=lambda d: d["reasons"].map(explain_reasons)
    )
    tooltip = [
        alt.Tooltip("t:Q", title="Reading"),
        alt.Tooltip("real_answer:N", title="Real answer"),
        alt.Tooltip("why:N", title="Why the AI raised an alarm"),
    ]

    panels = []
    for i, (key, title, scale) in enumerate(SERIES):
        y_axis = alt.Axis(format="~s", values=[100, 1_000, 10_000, 100_000]) if scale == "log" else alt.Axis()
        y = alt.Y(f"{key}:Q", title=title, scale=alt.Scale(type=scale, zero=False), axis=y_axis)
        px = x if i == len(SERIES) - 1 else x.axis(labels=False, title=None)
        layers = [
            alt.Chart(spans).mark_rect(color=TRUTH_COLOR, opacity=0.3).encode(
                x=alt.X("start:Q", scale=alt.Scale(domain=domain)), x2="end:Q"
            ),
            alt.Chart(df).mark_line(color=LINE_COLOR, strokeWidth=1.5).encode(x=px, y=y),
            alt.Chart(flagged).mark_circle(color=DETECT_COLOR, size=45, opacity=1).encode(x=px, y=y, tooltip=tooltip),
        ]
        if cursor is not None:  # mark the reading being reviewed
            layers.append(
                alt.Chart(pd.DataFrame({"t": [cursor]})).mark_rule(color=CURSOR_COLOR, strokeDash=[4, 3]).encode(
                    x=alt.X("t:Q", scale=alt.Scale(domain=domain))
                )
            )
        panels.append(alt.layer(*layers).properties(height=130))
    return alt.vconcat(*panels, spacing=6).resolve_scale(x="shared")


def readings_table(shown: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(shown[-TABLE_ROWS:][::-1])
    return pd.DataFrame({
        "Reading": df["t"],
        "Temp (°C)": df["Temperature"],
        "Humidity (%)": df["Humidity"],
        "Device": df["Status"],
        "Connection time": df["duration"],
        "Real answer": df["truth_type"].map(TYPE_NAMES),
        "AI said": df["anomaly"].map({True: "🚨 Alarm", False: "Normal"}),
        "Result": [outcome(t, a) for t, a in zip(df["truth_label"], df["anomaly"])],
        "Why": df["reasons"].map(explain_reasons),
    })


def shown_readings() -> tuple[list[dict], int | None]:
    """The readings to display (up to the reviewed one) and the cursor, clamped to what's still kept."""
    cursor = st.session_state.cursor
    if cursor is not None and (not sim.history or not oldest_t() <= cursor < newest_t()):
        cursor = st.session_state.cursor = None
    if cursor is None:
        return sim.recent(WINDOW), None
    end = cursor - oldest_t() + 1
    history = list(sim.history)
    return history[max(0, end - WINDOW):end], cursor


def render_stream():
    sb = sim.scoreboard()
    n, tp, fp, tn, fn = sb["readings"], sb["tp"], sb["fp"], sb["tn"], sb["fn"]
    m = st.columns(4)
    m[0].metric("Readings checked", f"{n:,}")
    m[1].metric("AI was right", pct((tp + tn) / n if n else None), help="Share of all readings where the AI's call matched the real answer.")
    m[2].metric("Real problems caught", f"{tp:,} of {tp + fn:,}", help="Readings that really were a problem, and how many of them the AI raised an alarm for.")
    m[3].metric("False alarms", f"{fp:,} of {fp + tn:,}", help="Normal readings where the AI raised an alarm anyway.")
    if sim.mode == "live" and sb["events"]:
        latency = "" if sb["mean_latency"] is None else f", on average {sb['mean_latency']:.1f} readings after it started"
        st.caption(f"Problems you added: the AI noticed **{sb['events_detected']} of {sb['events']}**{latency}.")

    shown, cursor = shown_readings()
    if not shown:
        st.info("Press **▶ Start** (or **Next ▶**) to begin.", icon=":material/play_circle:")
        return

    if cursor is not None:
        st.warning(
            f"You're looking back at reading **#{cursor}** (the newest is #{newest_t()}). "
            "Press **Next ▶** to go forward, or **▶ Start** to jump back to live. "
            "The scores above always count every reading so far.",
            icon=":material/history:",
        )

    r = shown[-1]
    verdict = outcome(r["truth_label"], r["anomaly"])
    ai_said = f"🚨 Alarm: {explain_reasons(r['reasons'])}" if r["anomaly"] else "Normal"
    with st.container(border=True):
        c = st.columns([1, 2, 2, 1])
        c[0].markdown(f"**{'Viewing' if cursor is not None else 'Latest'}: reading #{r['t']}**")
        c[1].markdown(f"Real answer: **{TYPE_NAMES[r['truth_type']]}**")
        c[2].markdown(f"AI said: **{ai_said}**")
        c[3].markdown(f"**{verdict}**")

    st.markdown(
        f"**Last {len(shown)} readings.** "
        f"<span style='color:{LINE_COLOR}'>━ Blue line</span> = the value reported · "
        f"<span style='background:{TRUTH_COLOR}55;padding:0 4px'>Orange band</span> = a real problem was happening · "
        f"<span style='color:{DETECT_COLOR}'>●</span> Red dot = the AI raised an alarm. "
        "Red dots on orange = caught; orange without red dots = missed; red dots without orange = false alarm.",
        unsafe_allow_html=True,
    )
    st.altair_chart(stream_chart(pd.DataFrame(shown), cursor), width="stretch")

    st.markdown("**Latest readings** (newest first)")
    st.dataframe(readings_table(shown), hide_index=True, width="stretch")


def stream_page():
    st.subheader("Watch a live stream")
    st.markdown("Readings arrive one at a time, like a real device, and the AI judges each one as it comes in.")

    st.radio(
        "Where do the readings come from?", list(MODE_LABELS), format_func=MODE_LABELS.get,
        key="mode", horizontal=True, on_change=apply_source_settings,
    )
    st.info(MODE_HELP[sim.mode].format(n=len(test_rows)), icon=":material/info:")

    running = st.session_state.running
    cursor = st.session_state.cursor
    at_oldest = not sim.history or (cursor is not None and cursor <= oldest_t())
    c = st.columns([1, 1, 1, 1, 3])
    c[0].button("⏸ Pause" if running else "▶ Start", key="start", on_click=toggle_running, type="primary", width="stretch")
    c[1].button("◀ Previous", key="previous", on_click=go_previous, width="stretch", disabled=running or at_oldest,
                help="Look back at the reading before (while paused).")
    c[2].button("Next ▶", key="next", on_click=go_next, width="stretch", disabled=running,
                help="Move forward one reading (while paused).")
    c[3].button("Restart", key="restart", on_click=restart, width="stretch",
                help="Clear everything and start again from the beginning.")
    c[4].slider("Speed (readings per second)", 1, 10, 4, key="speed")

    if sim.mode == "live":
        with st.container(border=True):
            st.markdown("**Add a problem.** Press a button: the next readings will be changed to look like this problem.")
            attacks = [k for k, s in INJECTORS.items() if s.truth_type != "fault"]
            faults = [k for k, s in INJECTORS.items() if s.truth_type == "fault"]
            st.caption("🔴 Cyber-attacks: *Replay* uses real attack readings from the data; *Synthetic* fakes the attack while keeping the room's climate.")
            cols = st.columns(len(attacks))
            for col, kind in zip(cols, attacks):
                spec = INJECTORS[kind]
                col.button(spec.label, key=f"inject_{kind}", help=spec.description, type="primary",
                           on_click=inject, args=(kind,), width="stretch")
            st.caption("🟠 Sensor faults: the sensor itself goes wrong. These are *not* in the training data at all.")
            cols = st.columns(len(attacks))
            for col, kind in zip(cols, faults):
                spec = INJECTORS[kind]
                col.button(spec.label, key=f"inject_{kind}", help=spec.description,
                           on_click=inject, args=(kind,), width="stretch")
            with st.expander("Problem options"):
                s1, s2 = st.columns([3, 1])
                s1.slider("Strength", 0.0, 1.0, 1.0, 0.1, key="intensity",
                          help="How strong the problem is. Low strength is harder to spot.")
                s2.number_input("How many readings (0 = default)", min_value=0, max_value=500, value=0, key="length")
                st.radio(
                    "Room climate", list(CLIMATE_LABELS), format_func=CLIMATE_LABELS.get, key="climate",
                    on_change=apply_source_settings, horizontal=True,
                    help="'Made-up' moves temperature and humidity into ranges the AI never saw for normal readings. "
                    "Use it to see how much the AI leans on climate: false alarms rise with no attack at all.",
                )

    live_view()

    history = sim.history_frame()
    if not history.empty:
        st.download_button("Download all readings (CSV)", history.to_csv(index=False), "iot_simulation.csv", "text/csv")


@st.fragment(run_every=TICK_SECONDS if st.session_state.running and st.session_state.get("page") == "stream" else None)
def live_view():
    # Only the fragment reruns each tick. Full reruns (button clicks) also pass through here,
    # so step only when a tick has actually elapsed.
    now = time.monotonic()
    if st.session_state.running and now - st.session_state.last_tick >= TICK_SECONDS * 0.8:
        for _ in range(int(st.session_state.speed * TICK_SECONDS)):
            sim.step()
        st.session_state.last_tick = now
    render_stream()


# =====================================================================
# Page 3: how it works
# =====================================================================
def about_page():
    meta = load_detector().metadata
    metrics = meta["metrics"]
    rows = meta["rows"]

    st.subheader("How it works")
    st.markdown("#### 1. The data")
    st.markdown(
        f"The AI learned from **{rows['after_dedup']:,} real readings** from a smart-home temperature and humidity "
        "sensor. Each reading has the sensor values (temperature, humidity, on/off) and details of the network "
        "traffic (how long the connection lasted, how much data was sent or lost). Every reading is labelled "
        "**normal**, **DoS attack** or **MITM attack**."
    )
    st.markdown(
        f"The readings were split in two: **{rows['train']:,} to learn from** and **{rows['test']:,} kept aside for "
        "testing**. The AI never sees the test readings while learning, so testing on them shows how it would do on "
        "readings it has never met. The *Real test data* stream and the examples on *Check a reading* use these test readings."
    )

    st.markdown("#### 2. Three checks on every reading")
    st.markdown("Each reading goes through three independent checks. If **any** of them is worried, the AI raises an alarm.")
    c = st.columns(3)
    boxes = [
        ("🔎 Attack recogniser", "Studied past examples of DoS and MITM attacks and learned what they look like. "
         "Gives an *attack likelihood* and a guess of which attack it is.", "Random Forest"),
        ("❓ Unusual-traffic detector", "Studied only normal readings. Flags anything that doesn't fit, so it can "
         "notice problems it has never seen before.", "Isolation Forest"),
        ("📏 Sensor sanity checks", "Simple rules learned from normal readings: is the value possible, did it jump "
         "too fast, has the sensor stopped changing?", "Rule-based"),
    ]
    for col, (title, body, tech) in zip(c, boxes):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.markdown(body)
            st.caption(f"Technique: {tech}")

    st.markdown("#### 3. The problems it looks for")
    st.markdown(
        """
| Problem | What it means | Real-world example |
|---|---|---|
| **DoS attack** (Denial of Service) | An attacker floods the device so it can't do its job. | The thermostat stops responding. |
| **MITM attack** (Man-in-the-Middle) | An attacker secretly sits between the device and the network, reading or changing data. | The app shows 21 °C while the room is really 28 °C. |
| **Sensor spike** | One reading jumps to an impossible value. | A loose wire or electrical glitch. |
| **Sensor drift** | Readings slowly creep away from the truth. | An ageing sensor. |
| **Sensor stuck** | The sensor keeps reporting exactly the same value. | A frozen sensor that still looks "online". |
"""
    )

    st.markdown("#### 4. How well it does (on the test readings)")
    c = st.columns(3)
    c[0].metric("Normal vs attack: correct", f"{metrics['binary']['report']['accuracy']:.1%}")
    c[1].metric("Names the right attack type", f"{metrics['attack_type']['report']['accuracy']:.1%}")
    c[2].metric("Unusual-traffic detector: correct", f"{metrics['novelty']['report']['accuracy']:.1%}")

    st.markdown("#### 5. Known limitation")
    st.warning(
        "In this dataset, all the attacks happened while the room was within a narrow temperature and humidity "
        "range. So the AI partly learned *\"this climate means attack\"* rather than only the attack itself. "
        "To see this, open *Watch a live stream*, choose *Simulated device*, and set **Room climate** to "
        "*Made-up*: false alarms rise even though nothing is wrong.",
        icon=":material/warning:",
    )
    with st.expander("Technical details"):
        st.markdown(
            f"Trained {meta['trained_at']}. Model inputs: {', '.join(f'`{f}`' for f in meta['features'])}.\n\n"
            "Columns deliberately **not** used, because they let the model cheat:\n\n"
            + "\n".join(f"- `{c}`: {why}" for c, why in meta["dropped_columns"].items())
        )


{"check": check_page, "stream": stream_page, "about": about_page}[page]()
