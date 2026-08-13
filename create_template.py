# Source - https://stackoverflow.com/a/50751932
# Posted by ImportanceOfBeingErnest, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-12, License - CC BY-SA 4.0
import os 
import json
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt


import numpy as np

def padded_polygon_mask(coords, polygon_shape, output_shape):
    """
    Shift polygon coordinates so a polygon defined in `polygon_shape`
    is centered inside `output_shape`, then rasterize it.

    coords: (N, 2) array in (row, col) order.
    """
    coords = np.asarray(coords, dtype=float).copy()

    dh = output_shape[0] - polygon_shape[0]
    dw = output_shape[1] - polygon_shape[1]

    if dh < 0 or dw < 0:
        raise ValueError("output_shape must be >= polygon_shape")

    top = dh // 2
    left = dw // 2

    # polygon2mask coordinates are (row, col)
    coords[:, 0] += top
    coords[:, 1] += left

    mask = polygon2mask(output_shape, coords)

    return mask, coords

def erode_mask(mask, pixels):
    """
    Erode a binary mask inward by `pixels` pixels.

    Parameters
    ----------
    mask : ndarray, shape (H, W)
        Binary polygon mask.

    pixels : float
        Erosion distance in pixels.

    Returns
    -------
    eroded : ndarray
        Binary uint8 mask.
    """
    mask = np.asarray(mask, dtype=bool)

    distance = distance_transform_edt(mask)

    eroded = distance > pixels

    return eroded.astype(np.uint8)


def bezier(points, num=200):
    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

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
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(
    n=5,
    scale=0.8,
    randomness=0.0,
    seed=None,
    rotation=None,
):
    """
    Generate a polygon by deforming a regular polygon.

    Parameters
    ----------
    n : int
        Number of vertices.

    scale : float
        Overall size. Returned coordinates lie approximately in [0, scale].

    randomness : float
        Amount of deviation from a regular polygon.

        0.0 -> perfectly regular polygon
        0.1 -> small irregularity
        0.3 -> moderate irregularity
        0.5 -> substantial irregularity
        1.0 -> very irregular

        The same parameter controls both:
            - angular irregularity
            - radial / edge-length irregularity

    seed : int or None
        Random seed.

    rotation : float or None
        Global polygon rotation in radians.
        If None, use a random rotation.

    Returns
    -------
    points : ndarray, shape (n, 2)
        Ordered polygon vertices.
    """

    rng = np.random.default_rng(seed)

    if rotation is None:
        rotation = rng.uniform(0, 2 * np.pi)

    # Center and radius before final scaling
    center = np.array([0.5, 0.5])
    R = 0.45

    # Regular polygon angular positions
    spacing = 2 * np.pi / n
    theta = rotation + np.arange(n) * spacing

    if randomness > 0:

        # ---------------------------------
        # Random angular perturbation
        # ---------------------------------
        angle_noise = rng.uniform(-1, 1, n)

        # Prevent a global rotation from being counted as randomness
        angle_noise -= angle_noise.mean()

        # Maximum displacement is a fraction of vertex spacing.
        # 0.45 keeps neighboring angular positions from swapping
        # when randomness <= 1.
        theta += (
            randomness
            * 0.45
            * spacing
            * angle_noise
        )

        # ---------------------------------
        # Random radial perturbation
        # ---------------------------------
        radius_noise = rng.uniform(-1, 1, n)

        # Prevent overall size from changing systematically
        radius_noise -= radius_noise.mean()

        r = R * (
            1
            + randomness
            * 0.45
            * radius_noise
        )

    else:
        r = np.full(n, R)

    # Polar -> Cartesian
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    points = np.column_stack([x, y])

    return points * scale

def save_as_json(x, y, a, out_file):
    output_json = {"a":a, "x":x, "y":y}
    with open(out_file, 'w') as f:
        output_text = json.dumps(output_json, indent=4)
        f.write(output_text)


def sample_points(coords, n, mindist, seed=None, max_attempt=100):
    """
    Randomly sample n points such that every pair is at least
    `mindist` apart.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Candidate (x, y) coordinates.

    n : int
        Number of points to sample.

    mindist : float
        Minimum Euclidean distance between any two selected points.

    seed : int or None
        Random seed.

    max_attempt : int
        Number of randomized attempts.

    Returns
    -------
    selected : ndarray, shape (n, 2)
    """
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
            else:
                s = np.asarray(selected)

                d = np.linalg.norm(s - p, axis=1)

                if np.all(d >= mindist):
                    selected.append(p)

            if len(selected) == n:
                return np.asarray(selected)

    # raise ValueError(
    #     f"Could not find {n} points at least {mindist} apart."
    # )
    return None


def draw_shapes(
    points,
    mindist,
    shape=(1024, 1024),
    shape_type="circle"
):
    """
    Draw filled circles or squares centered at `points`
    into a 2D binary NumPy array.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Shape centers in (x, y) pixel coordinates.

    mindist : float
        Circle radius, or square half-width, in pixels.

    shape : tuple
        Output mask shape as (height, width).

    shape_type : {"circle", "square"}
        Type of shape to draw.

    Returns
    -------
    mask : ndarray, dtype uint8
        Binary mask: 1 inside shapes, 0 outside.
    """
    points = np.asarray(points)
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)

    for y, x in points:
        if shape_type == "circle":
            region = (
                (xx - x) ** 2 +
                (yy - y) ** 2
                <= mindist ** 2
            )

        elif shape_type == "square":
            region = (
                (np.abs(xx - x) <= mindist) &
                (np.abs(yy - y) <= mindist)
            )

        else:
            raise ValueError("shape_type must be 'circle' or 'square'")

        mask |= region
    return mask.astype(np.uint8)

if __name__ == "__main__":
    silhouettes_folder = "./data/silhouettes"
    control_folder = "./data/control"
    os.makedirs(silhouettes_folder, exist_ok=True)
    os.makedirs(control_folder, exist_ok=True)
    output_shape = [1024, 1024]
    basedist = 125
    polygon_shape = [700, 700]

    rs = [0, 0.4, 0.8, 1]
    n_vs = [3, 4, 5, 6, 7, 8]
    n_ps = [3, 4, 5, 6]
    es = [0.1, 0.5, 1]
    rads = [0.1, 0.2, 0.4]
    for randomness in rs:
        for n_vertices in n_vs:
            for n_points in n_ps:
                for edgy in es:
                    for rad in rads:
                        mindist = int(basedist*3.0/n_points)
                        fname = f"img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad}.png"
 

                        c = [0, 0]
                        a = get_random_points(n=n_vertices, scale=0.8, randomness=randomness) + c
                        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
                        x = polygon_shape[0] * x
                        y = polygon_shape[1] * y
                        x = np.int16(x)
                        y = np.int16(y)

                        coords = np.stack([x, y], axis=1)
                        mask, coords = padded_polygon_mask(coords, polygon_shape, output_shape)
                        sampled_points = sample_points(coords, n=n_points, mindist=mindist*2.5)
                        if sampled_points is None:
                            continue
                        output_array = draw_shapes(sampled_points, mindist=mindist, shape=output_shape, shape_type="circle")
                        output_array[mask] = 0
                        plt.imsave(
                            os.path.join(silhouettes_folder, fname),
                            output_array,
                            cmap="gray",
                            vmin=0,
                            vmax=1
                        )

                        eroded_mask = erode_mask(mask, 5)
                        mask = mask - eroded_mask
                        output_array = output_array + mask
                        plt.imsave(
                            os.path.join(control_folder, fname),
                            output_array,
                            cmap="gray",
                            vmin=0,
                            vmax=1
                        )