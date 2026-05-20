#!/usr/bin/env python3
"""Generate a 2D map (PGM + YAML) matching the turtlebot3_world maze."""
import numpy as np
from PIL import Image
import os

RESOLUTION = 0.05  # m/pixel
WIDTH_M = 10.0
HEIGHT_M = 10.0
WIDTH_PX = int(WIDTH_M / RESOLUTION)
HEIGHT_PX = int(HEIGHT_M / RESOLUTION)
ORIGIN_X = -WIDTH_M / 2
ORIGIN_Y = -HEIGHT_M / 2

# Occupancy grid: 0=free, 100=occupied, -1=unknown
grid = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.int8)

def world_to_pixel(wx, wy):
    px = int((wx - ORIGIN_X) / RESOLUTION)
    py = int((wy - ORIGIN_Y) / RESOLUTION)
    return px, py

def draw_circle(grid, cx, cy, radius):
    """Draw occupied circle on grid."""
    for dy in range(-int(radius/RESOLUTION)-1, int(radius/RESOLUTION)+2):
        for dx in range(-int(radius/RESOLUTION)-1, int(radius/RESOLUTION)+2):
            if dx*dx + dy*dy <= (radius/RESOLUTION)**2:
                px = int(cx/RESOLUTION - ORIGIN_X/RESOLUTION) + dx
                py = int(cy/RESOLUTION - ORIGIN_Y/RESOLUTION) + dy
                if 0 <= px < WIDTH_PX and 0 <= py < HEIGHT_PX:
                    grid[py, px] = 100

# Outer walls (boundary)
wall_half = 0.1
for x in np.arange(ORIGIN_X, ORIGIN_X + WIDTH_M, RESOLUTION):
    px, py = world_to_pixel(x, ORIGIN_Y)
    if 0 <= py < HEIGHT_PX:
        for w in range(int(wall_half/RESOLUTION)):
            if 0 <= py+w < HEIGHT_PX: grid[py+w, px] = 100
    px, py = world_to_pixel(x, ORIGIN_Y + HEIGHT_M)
    if 0 <= py < HEIGHT_PX:
        for w in range(int(wall_half/RESOLUTION)):
            if 0 <= py-w < HEIGHT_PX: grid[py-w, px] = 100

for y in np.arange(ORIGIN_Y, ORIGIN_Y + HEIGHT_M, RESOLUTION):
    px, py = world_to_pixel(ORIGIN_X, y)
    if 0 <= px < WIDTH_PX:
        for w in range(int(wall_half/RESOLUTION)):
            if 0 <= px+w < WIDTH_PX: grid[py, px+w] = 100
    px, py = world_to_pixel(ORIGIN_X + WIDTH_M, y)
    if 0 <= px < WIDTH_PX:
        for w in range(int(wall_half/RESOLUTION)):
            if 0 <= px-w < WIDTH_PX: grid[py, px-w] = 100

# 9 cylinders (3x3 grid at 1.1m spacing, radius 0.15m)
for gx in [-1.1, 0, 1.1]:
    for gy in [-1.1, 0, 1.1]:
        draw_circle(grid, gx, gy, 0.18)

# Hexagon meshes (approximated as cylinders/circles)
# head at (3.5, 0) - scale 0.8 hexagon ~0.4m radius
draw_circle(grid, 3.5, 0, 0.45)
# left_hand at (1.8, 2.7) - scale 0.55 hexagon ~0.3m
draw_circle(grid, 1.8, 2.7, 0.35)
# right_hand at (1.8, -2.7)
draw_circle(grid, 1.8, -2.7, 0.35)
# left_foot at (-1.8, 2.7)
draw_circle(grid, -1.8, 2.7, 0.35)
# right_foot at (-1.8, -2.7)
draw_circle(grid, -1.8, -2.7, 0.35)

# Wall mesh body - rotated 90deg, at center
# Approximate as a line obstacle
for dy in range(-80, 80):
    px, py = world_to_pixel(0, dy * RESOLUTION)
    if 0 <= px < WIDTH_PX and 0 <= py < HEIGHT_PX:
        for w in range(6):
            if 0 <= px+w < WIDTH_PX: grid[py, px+w] = 100
            if 0 <= px-w < WIDTH_PX: grid[py, px-w] = 100

# Save PGM
os.makedirs(os.path.dirname(__file__) + '/../maps', exist_ok=True)
pgm_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map.pgm')
img = Image.fromarray(grid.astype(np.uint8), mode='L')
# Invert for PGM format (0=occupied, 254=free)
img_pgm = Image.fromarray(254 - grid.astype(np.uint8), mode='L')
img_pgm.save(pgm_path)

# Save YAML
yaml_path = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map.yaml')
yaml_content = f"""image: map.pgm
mode: trinary
resolution: {RESOLUTION}
origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"Map saved: {pgm_path} ({WIDTH_PX}x{HEIGHT_PX})")
print(f"YAML saved: {yaml_path}")
