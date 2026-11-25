"""evaluate.py scores the negative condition as "no illusion" and tracks its counts."""
import json
import evaluate as E


def _responses(tmp_path):
    data = {"illusory": {"stim_a.jpg": {"0": {"has_illusion": True, "n_objects": 4}}},
            "control": {"stim_b.jpg": {"0": {"has_illusion": False, "n_objects": 4}}},
            "negative": {"stim_c.jpg": {"0": {"has_illusion": True, "n_objects": 3}}}}   # model fooled
    p = tmp_path / "model.json"; p.write_text(json.dumps(data))
    gt = tmp_path / "gt"; gt.mkdir()
    for s in ("stim_a", "stim_b", "stim_c"):
        (gt / f"{s}.json").write_text(json.dumps({"n_inducers": 3}))
    return p, gt


def test_negative_counts_as_no_illusion(tmp_path):
    p, _ = _responses(tmp_path)
    m = E.compute_has_illusion_metrics(p)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 0)   # "yes" on the negative is a false positive


def test_negative_in_counting_stats(tmp_path):
    p, gt = _responses(tmp_path)
    m = E.compute_counting_metrics(p, E.load_ground_truth_counts(gt))
    assert m["class_stats"]["negative"]["total"] == 1
    assert m["class_stats"]["negative"]["correct"] == 0           # gt count is n_inducers + 1 = 4
    assert m["class_stats"]["illusory"]["correct"] == 1
