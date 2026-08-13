from pipeline import orchestrate
from pipeline.utils import OUT, OBJ, slug

KAN = [{"type": "kanizsa", "params": {"n_sides": 3}}]
FAKE = "tests.fakeprovider:FakeProvider"
FAKE_CN = "tests.fakeprovider_cn:FakeProviderCN"


def test_end_to_end_generates_with_provider(cfg):
    c = cfg(silhouettes=KAN,
            models={"image": {"provider": FAKE}},
            pairs={"source": "explicit", "n_per_silhouette": 1,
                   "explicit": [{"object": "novel obj", "background": "novel bg"}]},
            dress={"backends": ["composite"]}, output={"contact_sheet": False})
    orchestrate.run(c)
    assert (OBJ / f"{slug('novel obj')}.png").exists()      # provider generated the asset
    assert len(list((OUT / "comp").glob("*.png"))) == 1
    assert len(list((OUT / "gt").glob("*.json"))) == 1


def test_offline_loads_no_provider(cfg):
    # assets present (fixture fake assets) + invalid provider name -> must NOT be loaded
    c = cfg(silhouettes=KAN,
            models={"image": {"provider": "nope_not_real"}},
            pairs={"source": "explicit", "n_per_silhouette": 1,
                   "explicit": [{"object": "white disc", "background": "black bg"}]},
            dress={"backends": ["composite"]}, output={"contact_sheet": False})
    orchestrate.run(c)                                       # would ValueError if provider loaded
    assert len(list((OUT / "comp").glob("*.png"))) == 1


def test_controlnet_without_dress_warns_and_skips(cfg, capsys):
    c = cfg(silhouettes=KAN,
            models={"image": {"provider": FAKE}},
            pairs={"source": "explicit", "n_per_silhouette": 1,
                   "explicit": [{"object": "white disc", "background": "black bg"}]},
            dress={"backends": ["composite", "controlnet"]}, output={"contact_sheet": False})
    orchestrate.run(c)
    assert "WARNING" in capsys.readouterr().out
    assert len(list((OUT / "comp").glob("*.png"))) == 1     # composite still ran
    assert not (OUT / "cn").exists() or not list((OUT / "cn").glob("*.png"))


def test_controlnet_with_dress_runs(cfg):
    c = cfg(silhouettes=KAN,
            models={"image": {"provider": FAKE_CN}},
            pairs={"source": "explicit", "n_per_silhouette": 1,
                   "explicit": [{"object": "white disc", "background": "black bg"}]},
            dress={"backends": ["controlnet"]}, output={"contact_sheet": False})
    orchestrate.run(c)
    assert len(list((OUT / "cn").glob("*.png"))) == 1
