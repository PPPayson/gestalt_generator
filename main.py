# Source - https://stackoverflow.com/a/50751932
# Posted by ImportanceOfBeingErnest, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-12, License - CC BY-SA 4.0
#
# Silhouette generator (from PPPayson/gestalt_generator). Renders clean 1024x1024 masks with
# plt.imsave (no matplotlib margin) and, for each figure, a paired CONTROL mask that additionally
# draws the illusory polygon's outline (a visible-contour positive control).
# The __main__ loop is wrapped into generate() so it can be imported and seeded; `python main.py`
# reproduces the grid into data/silhouettes/ (illusory) and data/control/ (control).
import os
import json
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt


def padded_polygon_mask(coords, polygon_shape, output_shape):
    """Center a polygon defined in `polygon_shape` inside `output_shape`, then rasterize it."""
    coords = np.asarray(coords, dtype=float).copy()
    dh = output_shape[0] - polygon_shape[0]
    dw = output_shape[1] - polygon_shape[1]
    if dh < 0 or dw < 0:
        raise ValueError("output_shape must be >= polygon_shape")
    coords[:, 0] += dh // 2
    coords[:, 1] += dw // 2
    mask = polygon2mask(output_shape, coords)
    return mask, coords


def erode_mask(mask, pixels):
    """Erode a binary mask inward by `pixels` pixels (via distance transform)."""
    mask = np.asarray(mask, dtype=bool)
    return (distance_transform_edt(mask) > pixels).astype(np.uint8)


def bezier(points, num=200):
    bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """Create a curve through points *a*. *rad* steers control-point distance; *edgy* how edgy."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, randomness=0.0, seed=None, rotation=None):
    """Generate a polygon by deforming a regular polygon (see original docstring)."""
    rng = np.random.default_rng(seed)
    if rotation is None:
        rotation = rng.uniform(0, 2 * np.pi)
    center = np.array([0.5, 0.5]); R = 0.45
    spacing = 2 * np.pi / n
    theta = rotation + np.arange(n) * spacing
    if randomness > 0:
        angle_noise = rng.uniform(-1, 1, n); angle_noise -= angle_noise.mean()
        theta += randomness * 0.45 * spacing * angle_noise
        radius_noise = rng.uniform(-1, 1, n); radius_noise -= radius_noise.mean()
        r = R * (1 + randomness * 0.45 * radius_noise)
    else:
        r = np.full(n, R)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y]) * scale


def save_as_json(x, y, a, out_file):
    with open(out_file, 'w') as f:
        f.write(json.dumps({"a": a, "x": x, "y": y}, indent=4))


def sample_points(coords, n, mindist, seed=None, max_attempt=100):
    """Randomly sample n points at least `mindist` apart (or None)."""
    points = np.asarray(coords)
    rng = np.random.default_rng(seed)
    if n > len(points):
        raise ValueError("n cannot exceed the number of candidate points.")
    for _ in range(max_attempt):
        order = rng.permutation(len(points))
        selected = []
        for idx in order:
            p = points[idx]
            if not selected:
                selected.append(p)
            elif np.all(np.linalg.norm(np.asarray(selected) - p, axis=1) >= mindist):
                selected.append(p)
            if len(selected) == n:
                return np.asarray(selected)
    return None


def draw_shapes(points, mindist, shape=(1024, 1024), shape_type="circle"):
    """Draw filled circles (or squares) of radius `mindist` centered at `points` -> binary mask."""
    points = np.asarray(points)
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)
    for y, x in points:
        if shape_type == "circle":
            region = (xx - x) ** 2 + (yy - y) ** 2 <= mindist ** 2
        elif shape_type == "square":
            region = (np.abs(xx - x) <= mindist) & (np.abs(yy - y) <= mindist)
        else:
            raise ValueError("shape_type must be 'circle' or 'square'")
        mask |= region
    return mask.astype(np.uint8)


def generate(sil_dir="data/silhouettes", ctrl_dir="data/control",
             output_shape=(1024, 1024), polygon_shape=(700, 700), basedist=125,
             rs=(0, 0.4, 0.8, 1), n_vs=(3, 4, 5, 6, 7, 8), n_ps=(3, 4, 5, 6),
             es=(0.1, 0.5, 1), rads=(0.1, 0.2, 0.4), contour_px=5, seed=None):
    """Render the silhouette grid. Each figure produces an illusory mask (sil_dir) and a paired
    control mask (ctrl_dir) that additionally draws the polygon outline. Pass `seed` to reproduce."""
    output_shape = list(output_shape); polygon_shape = list(polygon_shape)
    os.makedirs(sil_dir, exist_ok=True); os.makedirs(ctrl_dir, exist_ok=True)
    count = 0
    for randomness in rs:
        for n_vertices in n_vs:
            for n_points in n_ps:
                for edgy in es:
                    for rad in rads:
                        mindist = int(basedist * 3.0 / n_points)
                        fname = f"img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad}.png"
                        a = get_random_points(n=n_vertices, scale=0.8, randomness=randomness, seed=seed)
                        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
                        x = np.int16(polygon_shape[0] * x)
                        y = np.int16(polygon_shape[1] * y)
                        coords = np.stack([x, y], axis=1)
                        mask, coords = padded_polygon_mask(coords, polygon_shape, output_shape)
                        sampled = sample_points(coords, n=n_points, mindist=mindist * 2.5, seed=seed)
                        if sampled is None:
                            continue
                        out = draw_shapes(sampled, mindist=mindist, shape=output_shape, shape_type="circle")
                        out[mask] = 0                                  # carve the illusory polygon
                        plt.imsave(os.path.join(sil_dir, fname), out, cmap="gray", vmin=0, vmax=1)
                        ring = mask - erode_mask(mask, contour_px)     # polygon outline
                        plt.imsave(os.path.join(ctrl_dir, fname), out + ring, cmap="gray", vmin=0, vmax=1)
                        count += 1
    return count


if __name__ == "__main__":
    generate()
