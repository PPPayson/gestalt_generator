# Source - https://stackoverflow.com/a/50751932
# Posted by ImportanceOfBeingErnest, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-12, License - CC BY-SA 4.0
#
# Silhouette generator. Renders clean 1024x1024 masks with
# plt.imsave (no matplotlib margin) and, for each figure, a paired CONTROL mask that additionally
# draws the illusory polygon's outline (a visible-contour positive control).

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils import data_utils


def generate(
        pos_dir="data/positive",
        neg_dir = "data/negative",
        ctrl_dir="data/control",
        output_shape=(1024, 1024), 
        polygon_shape=(700, 700), 
        basedist=125,
        rs=(0, 0.4, 0.8, 1), 
        n_vs=(3, 4, 5, 6, 7, 8), 
        n_ps=(3, 4, 5, 6),
        es=(0.1, 0.5, 1), 
        rads=(0.1, 0.2, 0.4), 
        contour_px=5, 
        shape_type='circle',
        seed=None
        ):

    """
    Render the silhouette grid. Each figure produces an illusory mask (sil_dir) and a paired
    control mask (ctrl_dir) that additionally draws the polygon outline. Pass `seed` to reproduce.
    """

    output_shape = list(output_shape)
    polygon_shape = list(polygon_shape)
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(ctrl_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    count = 0
    sampled_count = 0
    metadata = {}
    for randomness in rs:
        for n_vertices in n_vs:
            for n_points in n_ps:
                for edgy in es:
                    for rad in rads:
                        count += 1
                        mindist = int(basedist * 3.0 / n_points)
                        a = data_utils.get_random_points(n=n_vertices, scale=0.8, randomness=randomness, seed=seed)
                        x, y, _ = data_utils.get_bezier_curve(a, rad=rad, edgy=edgy)
                        x = np.int16(polygon_shape[0] * x)
                        y = np.int16(polygon_shape[1] * y)
                        coords = np.stack([x, y], axis=1)
                        mask, coords = data_utils.padded_polygon_mask(coords, polygon_shape, output_shape)
                        sampled = data_utils.sample_points(coords, n=n_points, mindist=mindist * 2.5, seed=seed)
                        if sampled is None:
                            continue
                        out = data_utils.draw_shapes(sampled, mindist=mindist, shape=output_shape, shape_type=shape_type)
                        out[mask] = 0 # carve the illusory polygon
                        ring = mask - data_utils.erode_mask(mask, contour_px) # polygon outline
                        neg_out = data_utils.shuffle_shapes(out, sampled, mindist=mindist,shape=(1024, 1024), shape_type=shape_type)
                        if neg_out is None:
                            continue
                        fname = f"img_{str(sampled_count).zfill(5)}.png"
                        metadata[fname] = {
                            "randomness": randomness,
                            "n_vertices": n_vertices,
                            "n_points": n_points,
                            "edgy": edgy,
                            "rad": rad
                            }
                        plt.imsave(os.path.join(ctrl_dir, fname), out + ring, cmap="gray", vmin=0, vmax=1)
                        plt.imsave(os.path.join(pos_dir, fname), out, cmap="gray", vmin=0, vmax=1)
                        plt.imsave(os.path.join(neg_dir, fname), neg_out, cmap="gray", vmin=0, vmax=1)
                        sampled_count += 1
    print(count, sampled_count)
    with open(os.path.join(os.path.dirname(pos_dir), "metadata.json"), 'w') as f:
        metadata_json = json.dumps(metadata, indent=4)
        f.write(metadata_json)
    return count


if __name__ == "__main__":
    generate()
