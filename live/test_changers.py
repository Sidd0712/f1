import json
from pathlib import Path

import pytest

from live.changers import race_changers

CASES = json.loads((Path(__file__).resolve().parent.parent / "site/src/test/fixtures/changer_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_case(case):
    laps = {int(k): v for k, v in case["laps"].items()}
    assert race_changers(laps, case["upTo"]) == case["expected"]
