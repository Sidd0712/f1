"""Circuit outlines as SVG paths, from the MIT-licensed bacinger/f1-circuits GeoJSON."""

import json
import math
import unicodedata
from pathlib import Path

IDS = [
    ("albert park", "au-1953"), ("enzo e dino ferrari", "it-1953"), ("mugello", "it-1914"), ("monza", "it-1922"), ("hermanos rodriguez", "mx-1962"),
    ("algarve", "pt-2008"), ("jose carlos pace", "br-1940"), ("bahrain", "bh-2002"), ("baku", "az-2016"), ("villeneuve", "ca-1978"),
    ("zandvoort", "nl-1948"), ("paul ricard", "fr-1969"), ("catalunya", "es-1991"), ("monaco", "mc-1929"), ("spa-francorchamps", "be-1925"),
    ("americas", "us-2012"), ("hockenheim", "de-1932"), ("hungaroring", "hu-1986"), ("istanbul", "tr-2005"), ("jeddah", "sa-2021"),
    ("las vegas", "us-2023"), ("losail", "qa-2004"), ("lusail", "qa-2004"), ("madring", "es-2026"), ("marina bay", "sg-2008"),
    ("miami", "us-2022"), ("nurburgring", "de-1927"), ("red bull ring", "at-1969"), ("sepang", "my-1999"), ("shanghai", "cn-2004"),
    ("silverstone", "gb-1948"), ("sochi", "ru-2014"), ("suzuka", "jp-1962"), ("yas marina", "ae-2009"),
]


def _plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def circuit_id(name):
    n = _plain(name)
    return next((cid for key, cid in IDS if key in n), None)


def geojson_to_path(coords, width=300, height=170, pad=10):
    lat0 = math.radians(sum(c[1] for c in coords) / len(coords))
    pts = [(c[0] * math.cos(lat0), c[1]) for c in coords]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(max(xs) - min(xs), 1e-9), max(max(ys) - min(ys), 1e-9)
    k = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    ox = (width - k * span[0]) / 2 - k * min(xs)
    oy = (height - k * span[1]) / 2 + k * max(ys)
    seg = [f"{ox + k * x:.1f} {oy - k * y:.1f}" for x, y in pts]
    return "M" + " L".join(seg) + " Z"


def build_tracks(geo_dir, circuit_names):
    out, missing = {}, []
    for name in sorted(set(circuit_names)):
        cid = circuit_id(name)
        f = Path(geo_dir) / f"{cid}.geojson" if cid else None
        if not f or not f.exists():
            missing.append(name)
            continue
        geom = json.loads(f.read_text(encoding="utf8"))["features"][0]["geometry"]
        coords = geom["coordinates"][0] if geom["type"] == "MultiLineString" else geom["coordinates"]
        out[name] = geojson_to_path(coords)
    return out, missing
