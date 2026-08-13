import numpy as np
from PIL import Image
from pipeline import silhouettes as S
from pipeline.geometry import inducer_labels, inducers, target_size
from pipeline.utils import SIL


def test_target_size_preserves_aspect():
    assert target_size((1920, 1440)) == (1024, 768)
    assert target_size((1024, 1024)) == (1024, 1024)


def test_inducer_labels_counts_and_canvas():
    S.produce_kanizsa(n_sides=4)
    lab, n, (w, h) = inducer_labels(SIL / "kanizsa_square.png")
    assert n == 4 and (w, h) == (1024, 1024)


def test_polarity_agnostic(tmp_path):
    # white inducers on black field (inverse of kanizsa) must still detect
    im = Image.new("L", (600, 600), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for cx, cy in [(150, 150), (450, 150), (300, 450)]:
        d.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=255)
    p = tmp_path / "inv.png"; im.save(p)
    assert len(inducers(p)) == 3
