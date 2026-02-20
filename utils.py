"""
@author: The Absolute Tinkerer
"""

import os
import math
import numpy as np

from PyQt5.QtGui import QColor


# ---------------------------------------------------------------------------
# Color Palettes  (#4)
# ---------------------------------------------------------------------------
# Each palette is a list of (hue, saturation, value) tuples in HSV [0-360, 0-255, 0-255].
# The renderer interpolates between these stops along the flow line.

COLOR_PALETTES = {
    'ocean':    [(200, 220, 255), (220, 180, 240), (180, 240, 200), (160, 200, 255)],
    'sunset':   [(10, 240, 255),  (30, 220, 255),  (50, 200, 240),  (340, 230, 220)],
    'forest':   [(100, 200, 200), (130, 230, 180), (80, 180, 230),  (60, 160, 250)],
    'neon':     [(280, 255, 255), (320, 240, 255), (180, 255, 240), (60, 255, 255)],
    'ember':    [(0, 250, 255),   (20, 240, 240),  (40, 200, 255),  (350, 230, 200)],
    'ice':      [(190, 160, 255), (210, 200, 240), (230, 180, 255), (170, 140, 250)],
    'aurora':   [(120, 220, 240), (160, 240, 255), (280, 200, 230), (80, 180, 250)],
    'monochrome': [(0, 0, 240),   (0, 0, 200),     (0, 0, 160),     (0, 0, 120)],
}


def palette_color(palette_name, t, alpha=20):
    """
    Interpolate a color from a named palette.

    Parameters:
    -----------
    palette_name : str
        Key into COLOR_PALETTES
    t : float or numpy array
        Position along the palette, 0.0 to 1.0
    alpha : int
        Alpha value for the returned color(s)

    Returns:
    --------
    If t is scalar: a single QColor
    If t is array:  tuple of (hue_array, sat_array, val_array) for vectorized use
    """
    stops = COLOR_PALETTES.get(palette_name, COLOR_PALETTES['ocean'])
    n = len(stops)

    if isinstance(t, np.ndarray):
        t = np.clip(t, 0.0, 1.0)
        idx_f = t * (n - 1)
        idx0 = np.floor(idx_f).astype(int)
        idx1 = np.minimum(idx0 + 1, n - 1)
        frac = idx_f - idx0

        h = np.zeros_like(t)
        s = np.zeros_like(t)
        v = np.zeros_like(t)
        for i, (sh, ss, sv) in enumerate(stops):
            mask0 = idx0 == i
            mask1 = idx1 == i
            h[mask0] += sh * (1 - frac[mask0])
            h[mask1] += stops[min(i, n-1)][0] * frac[mask1] if np.any(mask1) else 0
            s[mask0] += ss * (1 - frac[mask0])
            s[mask1] += stops[min(i, n-1)][1] * frac[mask1] if np.any(mask1) else 0
            v[mask0] += sv * (1 - frac[mask0])
            v[mask1] += stops[min(i, n-1)][2] * frac[mask1] if np.any(mask1) else 0

        # Simpler correct interpolation
        h0 = np.array([stops[i][0] for i in idx0], dtype=np.float64)
        s0 = np.array([stops[i][1] for i in idx0], dtype=np.float64)
        v0 = np.array([stops[i][2] for i in idx0], dtype=np.float64)
        h1 = np.array([stops[i][0] for i in idx1], dtype=np.float64)
        s1 = np.array([stops[i][1] for i in idx1], dtype=np.float64)
        v1 = np.array([stops[i][2] for i in idx1], dtype=np.float64)

        h = h0 + frac * (h1 - h0)
        s = s0 + frac * (s1 - s0)
        v = v0 + frac * (v1 - v0)
        return h % 360, np.clip(s, 0, 255), np.clip(v, 0, 255)
    else:
        t = max(0.0, min(1.0, t))
        idx_f = t * (n - 1)
        idx0 = int(idx_f)
        idx1 = min(idx0 + 1, n - 1)
        frac = idx_f - idx0
        h = stops[idx0][0] + frac * (stops[idx1][0] - stops[idx0][0])
        s = stops[idx0][1] + frac * (stops[idx1][1] - stops[idx0][1])
        v = stops[idx0][2] + frac * (stops[idx1][2] - stops[idx0][2])
        return QColor_HSV(h % 360, max(0, min(255, s)), max(0, min(255, v)), alpha)


def QColor_HSV(h, s, v, a=255):
    """
    Hue        : > -1 [wraps between 0-360]
    Saturation : 0-255
    Value      : 0-255
    Alpha      : 0-255
    """
    color = QColor()
    color.setHsv(*[int(e) for e in [h, s, v, a]])
    return color


def save(p, fname='image', folder='Images', extension='jpg', quality=100, overwrite=True):
    if not os.path.exists(folder):
        os.mkdir(folder)

    # The image name
    imageFile = f'{folder}/{fname}.{extension}'

    # Do not overwrite the image if it exists already
    if os.path.exists(imageFile):
        assert overwrite, 'File exists and overwrite is set to False!'

    # fileName, format, quality [0 through 100]
    p.saveImage(imageFile, imageFile[-3:], quality)


def Perlin2D(width, height, n_x, n_y, clampHorizontal=False, clampVertical=False):
    """
    Constructor

    Optimizations were gained from studying:
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py

    Parameters:
    -----------
    width : int
        The width of the canvas
    height : int
        The height of the canvas
    n_x : int
        The number of x tiles; must correspond to an integer x-edge length
    n_y : int
        The number of y tiles; must correspond to an integer y-edge length
    clampHorizontal : boolean
        Imagine the Perlin Noise on a sheet of paper - form a cylinder with
        the horizontal edges. If True, cylinder will be continuous noise
    clampVertical : boolean
        Imagine the Perlin Noise on a sheet of paper - form a cylinder with
        the vertical edges. If True, cylinder will be continuous noise

    Returns:
    --------
    <value> : numpy array
        noise values for array[width, height] between -1 and 1
    """
    # First ensure even number of n_x and n_y divide into the width and height,
    # respectively
    msg = 'n_x and n_y must evenly divide into width and height, respectively'
    assert width % n_x == 0 and height % n_y == 0, msg

    # We start off by defining our interpolation function
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Next, we generate the gradients that we are using for each corner point
    # of the grid
    angles = 2 * np.pi * np.random.rand(n_x + 1, n_y + 1)
    r = math.sqrt(2)  # The radius of the unit circle
    gradients = np.dstack((r * np.cos(angles), r * np.sin(angles)))

    # Now, if the user has chosen to clamp at all, set the first and last row/
    # column equal to one another
    if clampHorizontal:
        gradients[-1, :] = gradients[0, :]
    if clampVertical:
        gradients[:, -1] = gradients[:, 0]

    # Now that gradient vectors are complete, we need to create the normalized
    # distance from each point to its starting grid point. In other words, this
    # is the normalized distance from the grid tile's origin based upon the
    # grid tile's width and height
    delta = (n_x / width, n_y / height)
    grid = np.mgrid[0:n_x:delta[0], 0:n_y:delta[1]].transpose(1, 2, 0) % 1

    # At this point, we need to compute the dot products for each corner of the
    # grid. To do this, we first need proper-dimensioned gradient vectors - do
    # this now. A computation for number of points per tile is needed as well
    px, py = int(width / n_x), int(height / n_y)
    gradients = gradients.repeat(px, 0).repeat(py, 1)
    g00 = gradients[:-px, :-py]
    g10 = gradients[px:, :-py]
    g01 = gradients[:-px, py:]
    g11 = gradients[px:, py:]

    # Compute dot products for each corner
    d00 = np.sum(g00 * grid, 2)
    d10 = np.sum(g10 * np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])), 2)
    d01 = np.sum(g01 * np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)), 2)
    d11 = np.sum(g11 * np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)), 2)

    # We're doing improved perlin noise, so we use a fade function to compute
    # the x and y fractions used in the linear interpolation computation
    # t is the faded grid
    # u is the faded dot product between the top corners
    # v is the faded dot product between the bottom corners
    # _x and _y are the fractional (0-1) location of x, y in the tile
    t = fade(grid)
    u = d00 + t[:, :, 0] * (d10 - d00)
    v = d01 + t[:, :, 0] * (d11 - d01)

    # Now perform the second dimension's linear interpolation to return value
    return u + t[:, :, 1] * (v - u)


def FractalPerlin2D(width, height, n_x, n_y, octaves=6, persistence=0.5,
                    lacunarity=2, clampHorizontal=False, clampVertical=False):
    """
    Multi-octave fractal Perlin noise (fBm) for richer, more organic patterns.

    Layers multiple frequencies of Perlin noise, each with decreasing amplitude
    and increasing frequency, producing turbulent detail at every scale.

    Parameters:
    -----------
    width : int
        The width of the canvas
    height : int
        The height of the canvas
    n_x : int
        Base number of x tiles for the lowest frequency octave
    n_y : int
        Base number of y tiles for the lowest frequency octave
    octaves : int
        Number of noise layers to combine (more = finer detail)
    persistence : float
        Amplitude decay per octave (0.5 = each octave is half as strong)
    lacunarity : int
        Frequency multiplier per octave (2 = each octave doubles frequency)
    clampHorizontal : boolean
        If True, noise wraps seamlessly along horizontal edges
    clampVertical : boolean
        If True, noise wraps seamlessly along vertical edges

    Returns:
    --------
    <value> : numpy array
        Noise values for array[width, height], normalized between -1 and 1
    """
    noise = np.zeros((width, height), dtype=np.float64)
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        freq_x = n_x
        freq_y = n_y

        # Ensure frequency tiles divide evenly into canvas dimensions
        # by snapping to the nearest valid divisor
        while width % freq_x != 0 and freq_x > 1:
            freq_x -= 1
        while height % freq_y != 0 and freq_y > 1:
            freq_y -= 1

        octave_noise = Perlin2D(width, height, freq_x, freq_y,
                                clampHorizontal, clampVertical)
        noise += amplitude * octave_noise
        max_amplitude += amplitude

        amplitude *= persistence
        n_x *= lacunarity
        n_y *= lacunarity

    # Normalize to [-1, 1]
    noise /= max_amplitude
    return noise


# ---------------------------------------------------------------------------
# 1. Domain Warping
# ---------------------------------------------------------------------------

def DomainWarp(width, height, base_noise_fn, warp_strength=100.0, warp_octaves=4):
    """
    Distort the coordinate space itself before sampling noise, producing
    swirling, turbulent structures impossible with plain Perlin.

    Instead of sampling noise(x, y), we sample:
        noise(x + warp_strength * fbm1(x,y), y + warp_strength * fbm2(x,y))

    Parameters:
    -----------
    width, height : int
        Canvas dimensions
    base_noise_fn : callable(width, height) -> ndarray
        Function that returns a [-1, 1] noise field
    warp_strength : float
        Pixel displacement magnitude (higher = more distortion)
    warp_octaves : int
        Octaves for the warp offset fields

    Returns:
    --------
    ndarray : warped noise field, same shape as base, normalized to [-1, 1]
    """
    # Two independent noise fields drive the x and y coordinate offsets
    warp_x = FractalPerlin2D(width, height, 3, 3, octaves=warp_octaves,
                              persistence=0.5, lacunarity=2)
    warp_y = FractalPerlin2D(width, height, 3, 3, octaves=warp_octaves,
                              persistence=0.5, lacunarity=2)

    # Build the base noise field to sample from
    base = base_noise_fn(width, height)

    # Create warped coordinate grids
    gx, gy = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    wx = np.clip((gx + warp_strength * warp_x).astype(int), 0, width - 1)
    wy = np.clip((gy + warp_strength * warp_y).astype(int), 0, height - 1)

    warped = base[wx, wy]
    # Renormalize
    mn, mx = warped.min(), warped.max()
    if mx - mn > 0:
        warped = 2.0 * (warped - mn) / (mx - mn) - 1.0
    return warped


# ---------------------------------------------------------------------------
# 2. Curl Noise
# ---------------------------------------------------------------------------

def CurlNoise2D(noise_field, scale=1.0):
    """
    Derive a divergence-free (incompressible) 2D velocity field from a scalar
    noise field by taking the curl of a potential function.

    curl(psi) = (d_psi/dy, -d_psi/dx)

    This guarantees no sources or sinks, producing smooth, fluid-like flow
    with full surface coverage — particles never converge to a point.

    Parameters:
    -----------
    noise_field : ndarray (W, H)
        Scalar potential field (e.g. from FractalPerlin2D or DomainWarp)
    scale : float
        Multiplier on the resulting velocity vectors

    Returns:
    --------
    (vx, vy) : tuple of ndarray (W, H)
        The curl-derived velocity components
    """
    # Central differences: d_psi/dx and d_psi/dy
    dpsi_dx = np.zeros_like(noise_field)
    dpsi_dy = np.zeros_like(noise_field)

    dpsi_dx[1:-1, :] = (noise_field[2:, :] - noise_field[:-2, :]) / 2.0
    dpsi_dy[:, 1:-1] = (noise_field[:, 2:] - noise_field[:, :-2]) / 2.0

    # Boundary: forward/backward difference
    dpsi_dx[0, :] = noise_field[1, :] - noise_field[0, :]
    dpsi_dx[-1, :] = noise_field[-1, :] - noise_field[-2, :]
    dpsi_dy[:, 0] = noise_field[:, 1] - noise_field[:, 0]
    dpsi_dy[:, -1] = noise_field[:, -1] - noise_field[:, -2]

    # Curl: rotate gradient 90 degrees
    vx = scale * dpsi_dy
    vy = scale * (-dpsi_dx)
    return vx, vy


# ---------------------------------------------------------------------------
# 5. Attractor / Repulsor Points
# ---------------------------------------------------------------------------

def make_attractors(width, height, n_attractors=3, n_repulsors=2, seed=None):
    """
    Generate random attractor and repulsor points for compositional structure.

    Returns:
    --------
    list of (x, y, strength) where strength > 0 is attractor, < 0 is repulsor
    """
    rng = np.random.RandomState(seed)
    points = []
    # Attractors: positive strength pulls particles inward
    for _ in range(n_attractors):
        x = rng.randint(int(width * 0.15), int(width * 0.85))
        y = rng.randint(int(height * 0.15), int(height * 0.85))
        strength = rng.uniform(0.3, 1.0)
        points.append((x, y, strength))
    # Repulsors: negative strength pushes particles outward
    for _ in range(n_repulsors):
        x = rng.randint(int(width * 0.1), int(width * 0.9))
        y = rng.randint(int(height * 0.1), int(height * 0.9))
        strength = rng.uniform(-1.0, -0.3)
        points.append((x, y, strength))
    return points


def apply_attractors(xs, ys, vx, vy, attractors, influence_radius=500.0):
    """
    Blend attractor/repulsor forces into velocity vectors (vectorized).

    For each particle, each attractor/repulsor applies a radial force
    that falls off with distance, nudging flow toward compositional
    focal points.

    Parameters:
    -----------
    xs, ys : ndarray
        Current particle positions
    vx, vy : ndarray
        Current velocity components (modified in place conceptually)
    attractors : list of (x, y, strength)
        From make_attractors()
    influence_radius : float
        Distance at which force drops to ~37% (1/e)

    Returns:
    --------
    (vx_new, vy_new) : modified velocity arrays
    """
    ax = np.zeros_like(xs)
    ay = np.zeros_like(ys)
    for (px, py, strength) in attractors:
        dx = px - xs
        dy = py - ys
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        falloff = np.exp(-dist / influence_radius)
        # Normalize direction, scale by strength and falloff
        ax += strength * falloff * dx / dist
        ay += strength * falloff * dy / dist
    return vx + ax, vy + ay
