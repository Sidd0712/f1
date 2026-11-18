"""Race-changer rule; mirrors raceChangers in site/src/replay/logic.ts (shared cases in changer_cases.json)."""


def _sc_start(laps, n):
    return bool(laps.get(n, {}).get("sc")) and not laps.get(n - 1, {}).get("sc")


def _event(laps, n):
    if _sc_start(laps, n):
        return "Safety car"
    prev, cur = laps.get(n - 1), laps.get(n)
    if not prev or not cur:
        return None
    for c in prev["cars"]:
        now = next((x for x in cur["cars"] if x["d"] == c["d"]), None)
        if c["pos"] is not None and c["pos"] <= 6 and (now is None or now["out"]):
            return f"{c['d']} retires"
    if any(c["pit"] and c["pos"] is not None and c["pos"] <= 4 for c in cur["cars"]):
        return "Pit stops"
    return None


def race_changers(laps, up_to):
    found = []
    nums = sorted(n for n in laps if n <= up_to and laps[n]["live"])

    def win(k, d):
        return next((x["win"] for x in laps.get(k, {}).get("live", []) if x["d"] == d), None)

    for n in nums:
        if found and found[-1]["lap"] + 5 > n:
            continue
        reason = _event(laps, n) or _event(laps, n - 1)
        if not reason:
            continue
        out = {c["d"] for c in laps[n]["cars"] if c["out"]}
        best = None
        for x in laps[n]["live"]:
            if x["d"] in out:
                continue
            for k in (n - 1, n - 2):
                frm = win(k, x["d"])
                if frm is None:
                    continue
                d = abs(x["win"] - frm)
                if d >= 0.15 and (best is None or d > abs(best["to"] - best["from"])):
                    best = {"lap": n, "driver": x["d"], "from": frm, "to": x["win"], "reason": reason}
        if best:
            found.append(best)
    return found
