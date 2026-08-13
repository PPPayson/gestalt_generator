from pipeline import contact, dress_composite as DC, silhouettes as S
from pipeline.utils import SIL, OUT


def test_contact_builds_sheet(cfg):
    S.produce_kanizsa(n_sides=3)
    metas = [{"stem": "kanizsa_triangle", "illusion": "kanizsa_triangle", "illusory_shape": "triangle"}]
    pairs = {"kanizsa_triangle": [{"object": "white disc", "background": "black bg", "natural": False}]}
    DC.run(cfg(), metas, pairs)
    contact.run(cfg())
    assert (OUT / "contact_sheet.png").exists()


def test_contact_no_manifest_is_safe(cfg):
    # no manifest yet -> should not raise
    contact.run(cfg())
    assert not (OUT / "contact_sheet.png").exists()
