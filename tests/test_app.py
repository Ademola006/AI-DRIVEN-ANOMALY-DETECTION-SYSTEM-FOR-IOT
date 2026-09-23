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


def sim(at):
    return at.session_state["sim"]


def test_app_starts_empty(app):
    assert sim(app).t == 0
    assert any("Run stream" in i.value for i in app.info)


def test_step_and_inject(app):
    next(b for b in app.button if b.label == "Step").click().run()
    assert sim(app).t == 1

    app.button(key="inject_dos").click().run()
    assert not app.exception
    s = sim(app)
    assert s.t == 2 and s.events[-1]["kind"] == "dos"
    assert s.events[-1]["detected_at"] == 1


def test_settings_reach_the_detector(app):
    app.sidebar.slider[1].set_value(0.8).run()
    app.sidebar.checkbox[2].uncheck().run()
    det = sim(app).detector
    assert det.threshold == pytest.approx(0.8)
    assert det.use_sensor_checks is False


def test_switching_to_replay_resets(app):
    next(b for b in app.button if b.label == "Step").click().run()
    app.sidebar.radio(key="mode").set_value("replay").run()
    s = sim(app)
    assert s.mode == "replay" and s.t == 0
