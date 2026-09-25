import pytest

from iot_detector.features import DROPPED_COLUMNS, FEATURES, to_frame
from iot_detector.generator import NormalGenerator
from iot_detector.injectors import INJECTORS, InjectorBank
from iot_detector.sensor_checks import SensorMonitor

LIMITS = {
    "Temperature": {"min": 20.0, "max": 40.0, "max_jump": 5.0},
    "Humidity": {"min": 30.0, "max": 70.0, "max_jump": 20.0},
    "flatline_readings": 3,
}


def reading(**overrides):
    base = {
        "duration": 650,
        "destination_bytes": 52,
        "missed_bytes": 0,
        "dst_ip_bytes": 197,
        "Temperature": 28.0,
        "Humidity": 50.0,
        "Status": "OFF",
    }
    return {**base, **overrides}


def test_leaky_columns_are_not_features():
    assert not set(DROPPED_COLUMNS) & set(FEATURES)


def test_to_frame_rejects_missing_features():
    with pytest.raises(ValueError, match="missing"):
        to_frame({"duration": 1})


@pytest.mark.parametrize("climate", ["recorded", "synthetic"])
def test_generator_produces_valid_readings(reference, climate):
    gen = NormalGenerator(reference, seed=0, climate=climate)
    normals = reference[reference["Label"] == 0]
    for _ in range(200):
        r = gen.next()
        assert set(FEATURES) <= set(r)
        assert r["dst_ip_bytes"] in set(normals["dst_ip_bytes"])
        assert r["Status"] in {"ON", "OFF"}


def test_every_injector_runs_for_its_length(reference):
    bank = InjectorBank(reference, seed=0)
    for kind, spec in INJECTORS.items():
        inj = bank.start(kind, intensity=1.0)
        for _ in range(spec.default_length):
            assert inj.active
            out = bank.apply(inj, reading())
            assert set(FEATURES) <= set(out)
        assert not inj.active


def test_unknown_injector_raises(reference):
    with pytest.raises(ValueError):
        InjectorBank(reference).start("nope")


def test_stuck_freezes_sensors(reference):
    bank = InjectorBank(reference, seed=0)
    inj = bank.start("stuck")
    first = bank.apply(inj, reading(Temperature=28.0, Humidity=50.0))
    later = bank.apply(inj, reading(Temperature=31.0, Humidity=40.0))
    assert (later["Temperature"], later["Humidity"]) == (first["Temperature"], first["Humidity"])


def test_sensor_monitor_rules():
    m = SensorMonitor(LIMITS)
    assert m.update(reading()) == []
    assert "Temperature jump" in m.update(reading(Temperature=35.0))
    assert "Humidity out of range" in m.update(reading(Temperature=35.0, Humidity=90.0))

    m.reset()
    reasons = [m.update(reading()) for _ in range(5)]
    assert reasons[2] == [] and "sensors flatlined" in reasons[3]
