"""
@author: The Absolute Tinkerer
"""

import os
import math
import time
import random

import numpy as np

from PIL import Image

from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtCore import QPointF, QRect

import painter
from utils import (QColor_HSV, save, Perlin2D, FractalPerlin2D,
                   DomainWarp, CurlNoise2D, COLOR_PALETTES, palette_color,
                   make_attractors, apply_attractors)


def draw_white_noise(width, height, fname):
    assert not os.path.exists(fname), 'File already exists!'

    # Create a matrix of random values between zero and one
    pixels = np.random.random(size=(height, width))

    # Now modify the random values to be 0-255 (pixel color range)
    pixels = 255*pixels

    # The function to write the array of pixels to an image requires integers, not float values
    pixels = pixels.astype(np.uint8)

    # We choose to make random values grayscale, so each RGB element is identical. This code adds the third dimension
    # to our pixels array
    pixels = pixels[:, :, np.newaxis]

    # We need to repeat each value to finalize the pixels arrays in the grayscale space
    pixels = np.repeat(pixels, 3, axis=2)

    # Now create the image from an array of pixels
    im = Image.fromarray(pixels)

    # Save the image to file
    im.save(fname)


def draw_perlin(nx, ny, width, height, fname):
    assert not os.path.exists(fname), 'File already exists'

    # Initialize Perlin Noise
    noise = (Perlin2D(width, height, nx, ny) + 1)/2

    # Convert to pixels
    pixels = 255 * noise
    pixels = pixels.astype(np.uint8)
    pixels = pixels[:, :, np.newaxis]
    pixels = np.repeat(pixels, 3, axis=2)

    # Create and save the image from pixels
    im = Image.fromarray(pixels)
    im.save(fname)

    return noise


def draw_vectors(nx, ny, width, height, seed=random.randint(0, 100000000), flow_length=100, n_vectors=50):
    p_path = f'{seed}_1_perlin_noise.jpg'
    v_path = f'{seed}_2_vectors'
    f_path = f'{seed}_3_flow_field'

    # Ensure we don't overwrite paths
    assert not os.path.exists(p_path), 'Perlin Noise image already exists!'
    assert not os.path.exists(v_path), 'Vectors image already exists!'
    assert not os.path.exists(f_path), 'Flow field image already exists!'

    # Set the random seed for repeatability
    np.random.seed(seed)

    # Create the Perlin Noise image
    noise = draw_perlin(nx, ny, width, height, p_path)

    # Initialize the painter object for drawing
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)  # allow smooth drawing

    def draw_arrow(p, x_i, y_i, length=100, angle=0):
        # Compute the second points and draw the arrow body
        x_f = x_i + length*math.cos(math.radians(angle))
        y_f = y_i - length*math.sin(math.radians(angle))
        p.drawLine(x_i, y_i, x_f, y_f)

        # Compute the arrow head second points
        a_angle1, a_angle2 = math.radians(angle-30), math.radians(angle+30)
        x1 = x_f - (length/10)*math.cos(a_angle1)
        y1 = y_f + (length/10)*math.sin(a_angle1)
        x2 = x_f - (length/10)*math.cos(a_angle2)
        y2 = y_f + (length/10)*math.sin(a_angle2)
        p.drawLine(x_f, y_f, x1, y1)
        p.drawLine(x_f, y_f, x2, y2)

    # Load the Perlin Noise image and draw it with the painter
    p.drawPixmap(QRect(0, 0, width, height), QPixmap(p_path))

    # Now we're drawing red arrows for vectors, so set the pen color to red
    p.setPen(QColor(255, 0, 0))

    # We need arrow locations, so create a grid of n_vectors x n_vectors, excluding the image border
    _nx, _ny = n_vectors, n_vectors
    dx, dy = width / (_nx + 1), height / (_ny + 1)
    x_points = [dx + i*dx for i in range(_nx)]
    y_points = [dy + i*dy for i in range(_ny)]

    # Draw the arrows
    for x in x_points:
        for y in y_points:
            angle = 360*noise[int(x), int(y)]
            draw_arrow(p, x, y, length=min(dx, dy), angle=angle)

    # Save the vector image
    save(p, fname=v_path, folder='.')

    # Now draw the flow field. Start by initializing a new painter
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)  # allow smooth drawing
    p.setPen(QColor(0, 0, 0))  # pen color set to black

    # Step size between points
    STEP_SIZE = 0.001 * max(width, height)

    # Draw the flow field
    for x in x_points:
        for y in y_points:
            # The starting position
            x_s, y_s = x, y
            # The current line length tracking variable
            c_len = 0
            while c_len < flow_length:
                # angle between 0 and 2*pi
                angle = 2 * noise[int(x_s), int(y_s)] * math.pi

                # Compute the new point
                x_f = x_s + STEP_SIZE * math.cos(angle)
                y_f = y_s - STEP_SIZE * math.sin(angle)

                # Draw the line
                p.drawLine(QPointF(x_s, y_s), QPointF(x_f, y_f))

                # Update the line length
                c_len += math.sqrt((x_f - x_s) ** 2 + (y_f - y_s) ** 2)

                # Break from the loop if the new point is outside our image bounds
                # or if we've exceeded the line length; otherwise update the point
                if x_f < 0 or x_f >= width or y_f < 0 or y_f >= height or c_len > flow_length:
                    break
                else:
                    x_s, y_s = x_f, y_f
    save(p, fname=f_path, folder='.')


def draw_flow_field(width, height, seed=random.randint(0, 100000000)):
    # Set the random seed for repeatability
    np.random.seed(seed)

    # These are color hues
    colors = [200, 140, 70, 340, 280]
    for i, mod in enumerate(colors):
        print('Starting Image %s/%s' % (i + 1, len(colors)))
        p = painter.Painter(width, height)

        # Allow smooth drawing
        p.setRenderHint(p.Antialiasing)

        # Draw the background color
        p.fillRect(0, 0, width, height, QColor(0, 0, 0))

        # Set the pen color
        p.setPen(QPen(QColor(150, 150, 225, 5), 2))

        num = 1
        for j in range(num):
            print('Creating Noise... (%s/%s)' % (j + 1, num))
            p_noise = Perlin2D(width, height, 2, 2)
            print('Noise Generated! (%s/%s)' % (j + 1, num))

            MAX_LENGTH = 2 * width
            STEP_SIZE = 0.001 * max(width, height)
            NUM = int(width * height / 1000)
            POINTS = [(random.randint(0, width - 1), random.randint(0, height - 1)) for i in range(NUM)]

            for k, (x_s, y_s) in enumerate(POINTS):
                print(f'{100 * (k + 1) / len(POINTS):.1f}'.rjust(5) + '% Complete', end='\r')

                # The current line length tracking variable
                c_len = 0

                # Actually draw the flow field
                while c_len < MAX_LENGTH:
                    # Set the pen color for this segment
                    sat = 200 * (MAX_LENGTH - c_len) / MAX_LENGTH
                    hue = (mod + 130 * (height - y_s) / height) % 360
                    p.setPen(QPen(QColor_HSV(hue, sat, 255, 20), 2))

                    # angle between -pi and pi
                    angle = p_noise[int(x_s), int(y_s)] * math.pi

                    # Compute the new point
                    x_f = x_s + STEP_SIZE * math.cos(angle)
                    y_f = y_s + STEP_SIZE * math.sin(angle)

                    # Draw the line
                    p.drawLine(QPointF(x_s, y_s), QPointF(x_f, y_f))

                    # Update the line length
                    c_len += math.sqrt((x_f - x_s) ** 2 + (y_f - y_s) ** 2)

                    # Break from the loop if the new point is outside our image bounds
                    # or if we've exceeded the line length; otherwise update the point
                    if x_f < 0 or x_f >= width or y_f < 0 or y_f >= height or c_len > MAX_LENGTH:
                        break
                    else:
                        x_s, y_s = x_f, y_f

            save(p, fname=f'image_{mod}_{num}_{seed}', folder='.', overwrite=True)


def draw_perlin_rounding(width, height, fname, seed=random.randint(0, 100000000)):
    # Ensure we don't overwrite paths
    assert not os.path.exists(fname), 'Image already exists!'

    # Set the random seed for repeatability
    np.random.seed(seed)

    # Initialize a new painter
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)

    # Draw the background color
    p.fillRect(0, 0, width, height, QColor(0, 0, 0))

    # Set the pen color
    p.setPen(QColor(200, 200, 200))

    print('Creating Noise...', end='', flush=True)
    noise = Perlin2D(width, height, 1, 1)
    print('Done!')

    # The maximum line length and step size
    MAX_LENGTH = 1000
    STEP_SIZE = 0.001 * max(width, height)

    # Compute a grid 200x200 points, centered in the screen
    dx, dy = width / (200 + 1), height / (200 + 1)
    POINTS = [[(i+1)*dx, (j+1)*dy] for i in range(200) for j in range(200)]

    for i, (x_s, y_s) in enumerate(POINTS):
        print(f'{100 * (i + 1) / len(POINTS):.1f}'.rjust(5) + '% Complete', end='\r')

        # The current line length tracking variable
        c_len = 0
        while c_len < MAX_LENGTH:
            # angle between -pi and pi
            angle = math.pi*noise[int(x_s), int(y_s)]

            # Round the angle to pi/4 increments
            angle = round(angle / (math.pi / 4)) * (math.pi / 4)

            # Compute the new point
            x_f = x_s + STEP_SIZE * math.cos(angle)
            y_f = y_s + STEP_SIZE * math.sin(angle)

            # Draw the line
            p.drawLine(x_s, y_s, x_f, y_f)

            # Update the line length
            c_len += math.sqrt((x_f - x_s) ** 2 + (y_f - y_s) ** 2)

            # Break from the loop if the new point is outside our image bounds
            # or if we've exceeded the line length; otherwise update the point
            if (x_f < 0 or x_f >= width or y_f < 0 or y_f >= height or
                    c_len > MAX_LENGTH):
                break
            else:
                x_s, y_s = x_f, y_f

    print('100% Complete!')
    save(p, fname=f'{fname}_{seed}', folder='.')


class Body:
    def __init__(self, x, y, vx, vy):
        self._position = np.array([x, y], dtype=np.float64)
        self._velocity = np.array([vx, vy], dtype=np.float64)

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    def update(self, dt):
        # update the body position
        self._position = self._position + dt*self._velocity


class ExpandingCircleRandom:
    def __init__(self, radius, num_bodies, center=(0, 0), v_limits=(-2, 2)):
        self._bodies = [Body(center[0] + radius*math.cos(i*2*math.pi/num_bodies),
                             center[1] + radius*math.sin(i*2*math.pi/num_bodies),
                             v_limits[0]+(v_limits[1]-v_limits[0])*random.random(),
                             v_limits[0]+(v_limits[1]-v_limits[0])*random.random()) for i in range(num_bodies)]

    def draw(self, dt, painter):
        # Connect the dots between each body
        for i in range(len(self._bodies)):
            # Handle the wrapping case
            if i == len(self._bodies) - 1:
                p1 = QPointF(*self._bodies[i].position)
                p2 = QPointF(*self._bodies[0].position)
            else:
                p1 = QPointF(*self._bodies[i].position)
                p2 = QPointF(*self._bodies[i+1].position)
            painter.drawLine(p1, p2)

        # Update the position of each body
        for i in range(len(self._bodies)):
            self._bodies[i].update(dt)


class ExpandingCircleNoise:
    def __init__(self, radius, num_bodies, noise, center=(0, 0), v_max=2):
        self._bodies = [Body(center[0] + radius*math.cos(i*2*math.pi/num_bodies),
                             center[1] + radius*math.sin(i*2*math.pi/num_bodies),
                             0, 0) for i in range(num_bodies)]
        self._v_max = v_max
        self._noise = noise

    def draw(self, dt, painter):
        # Connect the dots between each body
        for i in range(len(self._bodies)):
            # Handle the wrapping case
            if i == len(self._bodies) - 1:
                p1 = QPointF(*self._bodies[i].position)
                p2 = QPointF(*self._bodies[0].position)
            else:
                p1 = QPointF(*self._bodies[i].position)
                p2 = QPointF(*self._bodies[i + 1].position)
            painter.drawLine(p1, p2)

            # Try to update the velocity for each body. If we can't its because the point is beyond the noise
            # field we've created, so at that point, just maintain velocity.
            try:
                a = math.pi*self._noise[int(p1.x()), int(p1.y())]
                v = np.array([self._v_max*math.cos(a), self._v_max*math.sin(a)])
                self._bodies[i]._velocity = v
            except IndexError:
                pass

        # Update the position of each body
        for i in range(len(self._bodies)):
            self._bodies[i].update(dt)


def draw_delta_body(width, height, seed=random.randint(0, 100000000), mode='noise'):
    assert mode in ['noise', 'random'], 'Mode must either be "noise" or "random"'

    # Set the random seed for repeatability
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the painter
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)  # Allow smooth drawing

    # Draw the background color
    p.fillRect(0, 0, width, height, QColor(0, 0, 0))

    # Set the pen color
    p.setPen(QPen(QColor(220, 220, 220, 5), 1))

    # Initialize the expanding circle centered in the canvas
    if mode == 'random':
        circle = ExpandingCircleRandom(width/8, 100, center=(width/2, height/2), v_limits=(-2, 2))
    elif mode == 'noise':
        noise = Perlin2D(width, height, 5, 5)
        circle = ExpandingCircleNoise(width/6, 200, noise, center=(width/4, height/2), v_max=5)
    else:
        circle = None

    # Initialize the delta time we're applying to each update
    dt = 0.3

    iterations = 1000
    for i in range(iterations):
        circle.draw(dt, p)

    save(p, fname=f'delta_{mode}_{seed}', folder='.', overwrite=True)


def _render_flow_field_variant(args):
    """
    Render a single flow field variant using domain-warped curl noise.

    Pipeline per variant:
      1. Seed RNG uniquely so every image gets different structure
      2. Build a fractal Perlin potential field
      3. Domain-warp it (distort coordinate space for swirling forms)
      4. Derive curl noise velocity field (divergence-free, no sinks)
      5. Trace particles through the (vx, vy) field
      6. Draw with batched, color-grouped QPainter calls
    """
    (width, height, palette_name, seed, num_particles, max_length, step_size,
     noise_params, warp_strength) = args
    from PyQt5.QtWidgets import QApplication
    import sys

    _app = QApplication.instance() or QApplication(sys.argv)

    np.random.seed(seed)
    random.seed(seed)

    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)
    p.fillRect(0, 0, width, height, QColor(0, 0, 0))

    # --- #1 Domain Warping + #2 Curl Noise ---
    # Build the potential field: fractal Perlin warped through distorted coords
    octaves, persistence, lacunarity, base_nx, base_ny = noise_params

    def _base_noise(w, h):
        return FractalPerlin2D(w, h, base_nx, base_ny,
                               octaves=octaves, persistence=persistence,
                               lacunarity=lacunarity)

    if warp_strength > 0:
        potential = DomainWarp(width, height, _base_noise,
                               warp_strength=warp_strength, warp_octaves=4)
    else:
        potential = _base_noise(width, height)

    # Curl of the potential gives a divergence-free velocity field —
    # particles flow along iso-lines, never converging to a point
    vx_field, vy_field = CurlNoise2D(potential, scale=1.0)

    # --- #3 Variable Line Weight ---
    # Keep raw magnitude before normalizing — this drives pen width.
    # High curl gradient = thick lines, low = thin. Reveals field structure.
    mag = np.sqrt(vx_field**2 + vy_field**2)
    mag[mag < 1e-10] = 1e-10
    vx_field /= mag
    vy_field /= mag

    # Normalize magnitude to [0, 1] for weight lookup
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-10)

    # --- Particle tracing (vectorized) ---
    xs = np.random.randint(0, width, size=num_particles).astype(np.float64)
    ys = np.random.randint(0, height, size=num_particles).astype(np.float64)
    lengths = np.zeros(num_particles, dtype=np.float64)
    active = np.ones(num_particles, dtype=bool)

    BATCH_SIZE = 50

    while np.any(active):
        batch_segments = []

        for _ in range(BATCH_SIZE):
            if not np.any(active):
                break

            idx = np.where(active)[0]
            xi = np.clip(xs[idx].astype(np.int64), 0, width - 1)
            yi = np.clip(ys[idx].astype(np.int64), 0, height - 1)

            # Sample the curl velocity field directly — no angle conversion
            vx = vx_field[xi, yi]
            vy = vy_field[xi, yi]

            x_new = xs[idx] + step_size * vx
            y_new = ys[idx] + step_size * vy

            # --- #4 Palette-based color ---
            # Sample palette by vertical position; trail fade modulates sat
            color_t = ys[idx] / height
            hue, sat_base, val = palette_color(palette_name, color_t)
            # Fade saturation along the trail (bright head -> muted tail)
            trail_fade = (max_length - lengths[idx]) / max_length
            sat = sat_base * trail_fade

            # Line weight: blend of field magnitude and trail taper
            # mag_component: 0.5-1.0 based on curl strength at this point
            # taper: 1.0 at head, fading to 0.3 at tail
            trail_t = lengths[idx] / max_length
            mag_component = 0.5 + 0.5 * mag_norm[xi, yi]
            taper = 1.0 - 0.7 * trail_t
            weight = mag_component * taper  # range ~[0.15, 1.0]

            batch_segments.append((
                hue.copy(), sat.copy(), val.copy(),
                xs[idx].copy(), ys[idx].copy(),
                x_new.copy(), y_new.copy(),
                weight.copy()
            ))

            seg_len = np.sqrt((x_new - xs[idx])**2 + (y_new - ys[idx])**2)
            lengths[idx] += seg_len
            xs[idx] = x_new
            ys[idx] = y_new

            oob = (x_new < 0) | (x_new >= width) | (y_new < 0) | (y_new >= height)
            maxed = lengths[idx] > max_length
            active[idx[oob | maxed]] = False

        # Flush: group by quantized color + weight to reduce setPen calls
        for hue_arr, sat_arr, val_arr, x0, y0, x1, y1, wt_arr in batch_segments:
            hue_q = (hue_arr / 5).astype(np.int32) * 5
            sat_q = (sat_arr / 5).astype(np.int32) * 5
            val_q = (val_arr / 5).astype(np.int32) * 5
            wt_q = np.clip((wt_arr * 4).astype(np.int32), 0, 3)
            color_keys = hue_q * 100000 + sat_q * 1000 + val_q * 10 + wt_q

            for ck in np.unique(color_keys):
                mask = color_keys == ck
                h = int(hue_q[mask][0]) % 360
                s = max(0, min(255, int(sat_q[mask][0])))
                v = max(0, min(255, int(val_q[mask][0])))
                w = 0.5 + int(wt_q[mask][0])
                p.setPen(QPen(QColor_HSV(h, s, v, 20), w))

                for j in np.where(mask)[0]:
                    p.drawLine(QPointF(x0[j], y0[j]), QPointF(x1[j], y1[j]))

    traced = num_particles - int(np.sum(active))
    print(f'  {palette_name}: done ({traced}/{num_particles} particles traced)')

    fname = f'enhanced_flow_{palette_name}_{seed}'
    save(p, fname=fname, folder='Images', overwrite=True)
    return fname


def draw_flow_field_enhanced(width=7680, height=4320,
                             seed=None,
                             palettes=None,
                             num_particles=None,
                             max_length=None,
                             step_size=None,
                             octaves=6,
                             persistence=0.5,
                             lacunarity=2,
                             base_nx=2, base_ny=2,
                             warp_strength=100.0,
                             parallel=True):
    """
    Enhanced flow field generator with domain-warped curl noise.

    Noise pipeline: FractalPerlin -> DomainWarp -> CurlNoise -> (vx, vy)

    Parameters:
    -----------
    width, height : int
        Canvas dimensions (default: 7680x4320 for 8K)
    seed : int or None
        Random seed for reproducibility
    palettes : list of str
        Color palette names from COLOR_PALETTES for each variant.
        Available: ocean, sunset, forest, neon, ember, ice, aurora, monochrome
    num_particles : int or None
        Number of flow lines (None = width*height/500)
    max_length : int or None
        Max trace length per particle (None = 3*width)
    step_size : float or None
        Distance per trace step (None = 0.001*max(w,h))
    octaves : int
        Fractal noise octaves
    persistence : float
        Amplitude falloff per octave
    lacunarity : int
        Frequency multiplier per octave
    base_nx, base_ny : int
        Base Perlin tile counts
    warp_strength : float
        Domain warp displacement in pixels (0 = no warp, 100+ = heavy)
    parallel : bool
        Render color variants in parallel processes
    """
    import platform

    if seed is None:
        seed = random.randint(0, 100000000)
    if palettes is None:
        palettes = ['ocean', 'sunset', 'forest', 'neon', 'ember']
    if num_particles is None:
        num_particles = int(width * height / 500)
    if max_length is None:
        max_length = 3 * width
    if step_size is None:
        step_size = 0.001 * max(width, height)

    noise_params = (octaves, persistence, lacunarity, base_nx, base_ny)

    print(f'Enhanced Flow Field: {width}x{height}, {num_particles} particles, '
          f'{len(palettes)} variants, seed={seed}')
    print(f'Palettes: {palettes}')
    print(f'Noise: {octaves} octaves, persistence={persistence}, '
          f'lacunarity={lacunarity}, warp={warp_strength}px')

    task_args = []
    for i, pal in enumerate(palettes):
        variant_seed = seed + i * 7919
        task_args.append((width, height, pal, variant_seed, num_particles,
                          max_length, step_size, noise_params, warp_strength))

    t0 = time.time()

    if parallel and len(palettes) > 1:
        import multiprocessing as mp

        if platform.system() == 'Darwin':
            ctx = mp.get_context('fork')
        elif platform.system() == 'Windows':
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context()

        n_workers = min(len(palettes), ctx.cpu_count())
        print(f'Rendering {len(palettes)} variants in parallel '
              f'({n_workers} workers, {ctx.get_start_method()} mode)...')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_render_flow_field_variant, task_args)
    else:
        print(f'Rendering {len(palettes)} variants sequentially...')
        results = [_render_flow_field_variant(a) for a in task_args]

    elapsed = time.time() - t0
    print(f'All variants complete in {elapsed:.1f}s')
    print(f'Output files: {results}')
