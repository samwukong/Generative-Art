"""
@author: The Absolute Tinkerer
"""

import os
import random

from examples import (draw_flow_field, draw_delta_body, draw_white_noise,
                      draw_perlin, draw_vectors, draw_perlin_rounding,
                      draw_flow_field_enhanced)


if __name__ == '__main__':
    output_folder = 'Images'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # --- Enhanced flow field (domain-warped curl noise) ---
    # Pipeline: FractalPerlin -> DomainWarp -> CurlNoise -> (vx, vy)
    draw_flow_field_enhanced(
        width=7680,
        height=4320,
        palettes=['ocean', 'sunset', 'forest', 'neon', 'ember'],
        octaves=6,              # fractal detail layers
        persistence=0.5,        # amplitude decay per octave
        lacunarity=2,           # frequency doubling per octave
        warp_strength=100.0,    # domain warp displacement (px); 0=off
        parallel=True,          # render color variants across CPU cores
    )

    # --- Original examples (uncomment to use) ---
    # draw_flow_field(6000, 4000)
    # draw_white_noise(600, 300, f'{output_folder}/white_noise.jpg')
    # draw_perlin(5, 5, 1000, 1000, 'output_image.jpg')
    # draw_vectors(5, 5, 1000, 1000)
    # draw_perlin_rounding(6000, 4000, 'perlin_rounding')
    # draw_delta_body(2000, 2000, mode='noise')
