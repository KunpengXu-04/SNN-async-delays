"""One-off: crop the 3 step3 wad-vs-d0 raster panels down to wad-only,
near-square images for a readable 3-up figure (replaces the over-cramped
0.31\\textwidth x 3 layout of the original 2-panel images)."""
import os
from PIL import Image

SRC = "docs/enhanced/raster/4ops16k_K{k}_wad_vs_d0.png"
DST = "paper/figures/fig_step3_k{k}_wad_only.png"

for k in (1, 2, 3):
    src = SRC.format(k=k)
    im = Image.open(src)
    w, h = im.size
    print(k, "full size", w, h)
    # left panel only, drop the shared two-condition suptitle band at the top
    crop = im.crop((0, 45, w // 2 - 20, h))
    print("  cropped size", crop.size)
    dst = DST.format(k=k)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    crop.save(dst)
    print("  saved", dst)
