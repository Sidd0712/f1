import re

from site_export import tracks



def test_square_fits_box_with_padding():
    d = tracks.geojson_to_path([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
    xs, ys = nums[0::2], nums[1::2]
    assert d.startswith("M") and d.endswith("Z")
    assert min(xs) >= 10 and max(xs) <= 290 and min(ys) >= 10 and max(ys) <= 160


def test_y_is_flipped():
    d = tracks.geojson_to_path([[0, 0], [0, 10]])
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
    assert nums[1] > nums[3]


def test_every_circuit_has_an_id():
    names = ["Albert Park Grand Prix Circuit", "Autódromo José Carlos Pace", "Nürburgring", "Lusail International Circuit", "Losail International Circuit",
             "Las Vegas Strip Street Circuit", "Madring", "Circuit Park Zandvoort", "Autodromo Enzo e Dino Ferrari", "Sepang International Circuit"]
    assert all(tracks.circuit_id(n) for n in names)
    assert tracks.circuit_id("Autodromo Nazionale di Monza") == "it-1922"
    assert tracks.circuit_id("Nowhere Ring") is None
