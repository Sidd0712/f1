import base64
import json

import pandas as pd
import pytest

from live import push
from live.feed import to_master


def test_subscription_round_trip():
    sub = {"endpoint": "https://push.example/abc", "keys": {"p256dh": "x", "auth": "y"}}
    b64 = base64.urlsafe_b64encode(json.dumps(sub).encode()).decode().rstrip("=")
    assert push.parse_subscription(b64) == sub
    with pytest.raises(ValueError):
        push.parse_subscription(base64.urlsafe_b64encode(b'{"endpoint":"http://evil"}').decode())


def test_alert_text_spoiler_free_when_watching_later():
    c = {"lap": 34, "driver": "LEC", "from": 0.08, "to": 0.41, "reason": "Safety car"}
    assert push.alert_text(c, "Las Vegas GP", later=False) == ("Race changer · Las Vegas GP lap 34", "Safety car. LEC win chance 8% → 41%.")
    title, body = push.alert_text(c, "Las Vegas GP", later=True)
    assert "LEC" not in title + body and "%" not in title + body


def test_to_master_converts_timedeltas():
    laps = pd.DataFrame({"Driver": ["NOR"], "LapNumber": [3.0], "Position": [1.0], "LapTime": [pd.Timedelta(seconds=91.5)], "Time": [pd.Timedelta(seconds=300)],
                         "Stint": [1.0], "TyreLife": [3.0], "TrackStatus": ["1"], "PitInTime": [pd.NaT], "PitOutTime": [pd.NaT], "Compound": ["MEDIUM"]})
    d = to_master(laps, 2026, "Las Vegas Grand Prix")
    assert d.loc[0, "LapTime"] == 91.5 and d.loc[0, "Time"] == 300 and d.loc[0, "HasPit"] == 0 and d.loc[0, "Year"] == 2026
