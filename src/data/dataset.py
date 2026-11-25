"""Closure-sweep triplets read from a pipeline output folder (out/ by default).

For every silhouette stem that has all three files
    <root>/silhouettes/<stem>.png   illusory figure: bitten discs, no contour
    <root>/control/<stem>.png       the same figure with the polygon outline drawn
    <root>/negative/<stem>.png      the same discs shuffled so they share no contour
the dataset yields (ring, pos, neg): the contour-only outline (control minus illusory), the
illusory image and the negative, each a float32 0/1 array (then `transform`, if given).
Produce the folder with  python3 run.py configs/closure_sweep.yaml  (or any config)."""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class IllusionDataset(Dataset):
    def __init__(self, root="out", transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        dirs = [os.path.join(root, d) for d in ("silhouettes", "control", "negative")]
        for d in dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"{d} missing: run `python3 run.py configs/closure_sweep.yaml` first")
        stems = set(f[:-4] for f in os.listdir(dirs[0]) if f.endswith(".png"))
        for d in dirs[1:]:
            stems &= set(f[:-4] for f in os.listdir(d) if f.endswith(".png"))
        self.stems = sorted(stems)
        if not self.stems:
            raise RuntimeError(f"no silhouette with control + negative under {root}")

    def __len__(self):
        return len(self.stems)

    @staticmethod
    def _bin(path):
        return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        pos = self._bin(os.path.join(self.root, "silhouettes", f"{stem}.png"))
        ctrl = self._bin(os.path.join(self.root, "control", f"{stem}.png"))
        neg = self._bin(os.path.join(self.root, "negative", f"{stem}.png"))
        ring = (ctrl.astype(bool) & ~pos.astype(bool)).astype(np.uint8)
        out = [a.astype(np.float32) for a in (ring, pos, neg)]
        if self.transform is not None:
            out = [self.transform(a) for a in out]
        return tuple(out)
