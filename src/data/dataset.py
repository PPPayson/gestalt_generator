from torch.utils.data import Dataset
from src.utils import data_utils
import numpy as np


class IllusionDataset(Dataset):
    def __init__(
        self,
        rs=(0, 0.4, 0.8, 1),
        n_vs=(3, 4, 5, 6, 7, 8),
        n_ps=(3, 4, 5, 6),
        es=(0.1, 0.5, 1),
        rads=(0.1, 0.2, 0.4),
        output_shape = (1024, 1024),
        polygon_shape = (600, 600),
        basedist=100,
        contour_px=5,
        shape_type='circle',
        seed=0,
        transform=None,
        repeats=10
        ):
        super().__init__()
        self.contour_px = contour_px
        self.shape_type = shape_type
        self.output_shape = output_shape
        self.polygon_shape = polygon_shape
        self.basedist = basedist
        self.seed = seed
        self.transform=transform
        self.repeats=repeats
        dists = []
        for n_points in (n_ps):
            dists.append(int(self.basedist * 3.0)/ n_points)

        self.combinations = []
        for randomness in rs:
            for n_vertices in n_vs:
                for i, n_points in enumerate(n_ps):
                    for edgy in es:
                        for rad in rads:
                            for dist in dists[i:]:
                                self.combinations.append({
                                    'r': randomness,
                                    'n_v': n_vertices,
                                    'n_p': n_points,
                                    'edgy': edgy,
                                    'rad': rad,
                                    'dist': dist
                                })
        
    def __len__(self):
        return len(self.combinations) * self.repeats

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed+idx)
        idx = idx % len(self.combinations)

        combination = self.combinations[idx]
        randomness = combination['r']
        n_vertices = combination['n_v']
        n_points = combination['n_p']
        edgy = combination['edgy']
        rad = combination['rad']
        mindist = combination['dist']

        contour_return = None
        attempt = 0
        while contour_return is None:
            attempt += 1
            if attempt > 100:
                raise RuntimeError("Failed to load a valid sample after 100 attempts")
            a = data_utils.get_random_points(n=n_vertices, scale=0.8, randomness=randomness, rng=rng)
            x, y, _ = data_utils.get_bezier_curve(a, rad=rad, edgy=edgy)
            x = np.int16(self.polygon_shape[0] * x)
            y = np.int16(self.polygon_shape[1] * y)
            coords = np.stack([x, y], axis=1)
            mask, coords = data_utils.padded_polygon_mask(coords, self.polygon_shape, self.output_shape)
            sampled = data_utils.sample_points(coords, n=n_points, mindist=mindist * 2.5, rng=rng)
            if sampled is None:
                continue
            out = data_utils.draw_shapes(sampled, mindist=mindist, shape=self.output_shape, shape_type=self.shape_type)
            out[mask] = 0 # carve the illusory polygon
            neg_out = data_utils.shuffle_shapes(out, sampled, mindist=mindist,shape=self.output_shape, shape_type=self.shape_type, rng=rng)
            if neg_out is None:
                continue
            ring = mask - data_utils.erode_mask(mask, self.contour_px) # polygon outline
            contour_return = ring
            pos_return = out
            neg_return = neg_out
            if self.transform is not None:
                contour_return = self.transform(contour_return.astype(np.float32))
                pos_return = self.transform(pos_return.astype(np.float32))
                neg_return = self.transform(neg_return.astype(np.float32))
        return contour_return, pos_return, neg_return
        


            