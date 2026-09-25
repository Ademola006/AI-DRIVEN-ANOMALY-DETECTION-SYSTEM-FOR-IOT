import pytest

from iot_detector.simulator import Simulator


@pytest.fixture(scope="module")
def sim(model_dir):
    return Simulator(model_dir=model_dir, seed=0)


def test_recorded_normal_traffic_has_few_false_alarms(sim):
    sim.reset()
    sim.run(200)
    assert sim.scoreboard()["false_alarm_rate"] < 0.05


@pytest.mark.parametrize("kind", ["replay_dos", "dos", "mitm", "spike"])
def test_injected_attacks_are_detected(sim, kind):
    sim.reset()
    sim.run(10)
    sim.inject(kind, intensity=1.0)
    sim.run(20)
    events = sim.events_frame()
    assert events["detected"].iloc[0]
    assert events["latency"].iloc[0] <= 2


def test_injection_sets_ground_truth(sim):
    sim.reset()
    sim.inject("drift", length=5)
    h = sim.run(8)
    assert h["truth_label"].tolist() == [1] * 5 + [0] * 3
    assert set(h["truth_type"].iloc[:5]) == {"fault"}


def test_replay_uses_real_labels(sim):
    sim.mode = "replay"
    try:
        sim.reset()
        h = sim.run(50)
        test_rows = sim.reference[sim.reference["split"] == "test"].sort_values("row_id").head(50)
        assert h["truth_label"].tolist() == test_rows["Label"].tolist()
    finally:
        sim.mode = "live"
