from pathlib import Path

from live import serve


def resolve(path):
    h = object.__new__(serve.Handler)
    h.directory = str(serve.DIST)
    return Path(h.translate_path(path))


def test_data_paths_stay_inside_the_data_folder():
    for bad in ("/f1/data/../../../live/.keys/vapid_private.pem", "/f1/data/C:/Windows/win.ini", "/f1/data/%2e%2e/%2e%2e/%2e%2e/live/.keys/vapid_private.pem"):
        p = resolve(bad).resolve()
        assert serve.DATA.resolve() in p.parents or p == serve.DATA.resolve(), bad
        assert not p.exists() or p.is_relative_to(serve.DATA.resolve())


def test_normal_data_path_resolves():
    assert resolve("/f1/data/index.json") == (serve.DATA / "index.json").resolve()


def test_live_meta_uses_exported_grid_and_model_prior():
    import json
    idx = json.loads((serve.DATA / "index.json").read_text(encoding="utf8"))
    grid = [dict(driver=d, team=t, grid=i + 1) for i, (d, t) in enumerate([("NOR", "McLaren"), ("VER", "Red Bull Racing"), ("LEC", "Ferrari")])]
    idx["weekendGrid"] = dict(year=2026, round=22, grid=grid)
    meta = serve.live_meta(idx, 2026, 22, "Qatar Grand Prix", 57)
    assert meta["grid"][:3] == ["NOR", "VER", "LEC"]
    assert abs(sum(p["p"] for p in meta["prior"]) - 3) < 1e-3


def test_watching_later_hides_laps_from_the_state_file():
    class Fake:
        def snapshot(self):
            return {"laps": {"1": {"cars": []}}, "updatedAt": 1}
    st = serve.State()
    st.meta, st.race, st.later = {"race": "X"}, Fake(), True
    assert "laps" not in st.snapshot() or not st.snapshot()["laps"]
    st.later = False
    assert st.snapshot()["laps"]
