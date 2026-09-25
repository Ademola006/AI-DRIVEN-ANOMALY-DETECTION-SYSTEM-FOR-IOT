from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(model_dir):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


@pytest.fixture
def stream(app):
    app.radio(key="page").set_value("stream").run()
    assert not app.exception
    return app


def sim(at):
    return at.session_state["sim"]


def test_app_starts_on_check_page(app):
    assert sim(app).t == 0
    assert app.session_state["page"] == "check"
    assert any("Looks normal" in s.value or "Suspicious" in s.value for s in [*app.success, *app.error])


def test_manual_input_is_scored(app):
    app.button(key="example_dos").click().run()
    assert not app.exception
    assert app.session_state["example"]["kind"] == "dos"
    assert any("Real answer for this example: DoS attack" in i.value for i in app.info)

    app.number_input(key="in_Temperature").set_value(55.0).run()
    result = app.session_state["check_result"]
    assert result["anomaly"] and "Temperature out of range" in result["reasons"]
    assert not any("Real answer" in i.value for i in app.info)


def test_stream_starts_empty(stream):
    assert sim(stream).t == 0 and sim(stream).mode == "replay"
    assert any("Start" in i.value for i in stream.info)


def test_next_and_previous(stream):
    for _ in range(3):
        stream.button(key="next").click().run()
    assert sim(stream).t == 3

    stream.button(key="previous").click().run()
    stream.button(key="previous").click().run()
    assert stream.session_state["cursor"] == 0
    assert stream.button(key="previous").disabled  # at the oldest reading
    assert sim(stream).t == 3  # looking back does not change the stream

    stream.button(key="next").click().run()
    assert stream.session_state["cursor"] == 1
    stream.button(key="next").click().run()
    assert stream.session_state["cursor"] is None and sim(stream).t == 3  # back at the newest
    stream.button(key="next").click().run()
    assert sim(stream).t == 4  # now a new reading


def test_inject_in_simulated_mode(stream):
    stream.radio(key="mode").set_value("live").run()
    stream.button(key="next").click().run()
    assert sim(stream).t == 1

    stream.button(key="inject_dos").click().run()
    assert not stream.exception
    s = sim(stream)
    assert s.t == 2 and s.events[-1]["kind"] == "dos"
    assert s.events[-1]["detected_at"] == 1


def test_settings_reach_the_detector(app):
    app.slider(key="threshold").set_value(0.8).run()
    app.checkbox(key="use_sensor_checks").uncheck().run()
    det = sim(app).detector
    assert det.threshold == pytest.approx(0.8)
    assert det.use_sensor_checks is False


def test_switching_source_resets(stream):
    stream.button(key="next").click().run()
    stream.radio(key="mode").set_value("live").run()
    s = sim(stream)
    assert s.mode == "live" and s.t == 0


def test_about_page_renders(app):
    app.radio(key="page").set_value("about").run()
    assert not app.exception
