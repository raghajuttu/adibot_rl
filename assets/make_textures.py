"""Generate the procedural textures used by the assets (cardboard, wood).

    python assets/make_textures.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent / "textures"
rng = np.random.default_rng(7)


def smooth_noise(h, w, blur):
    img = Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255


def cardboard(size=512):
    # Kraft brown with paper fibres, blotches and faint corrugation ridges.
    base = np.array([176, 132, 86], dtype=np.float32)
    fibres = rng.normal(0, 1, (size, size)).astype(np.float32)
    fibres = np.asarray(Image.fromarray(((fibres * 20) + 128).clip(0, 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) - 128
    blotch = (smooth_noise(size, size, 18) - 0.5) * 50
    ridges = 6 * np.sin(np.arange(size) / size * 2 * np.pi * 48)[None, :]
    shade = fibres * 0.6 + blotch + ridges
    img = base[None, None, :] + shade[..., None] * np.array([1.0, 0.85, 0.65])
    return img.clip(0, 255).astype(np.uint8)


def wood(size=512):
    # Light oak: stretched noise for grain plus growth-ring bands.
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    warp = smooth_noise(size, size, 25) * 6
    rings = np.sin((y * 22 + warp) * np.pi) * 0.5 + 0.5
    grain = np.asarray(Image.fromarray((rng.random((size, size)) * 255).astype(np.uint8))
                       .resize((size, size // 16)).resize((size, size)), dtype=np.float32) / 255
    v = 0.75 + 0.15 * rings + 0.10 * grain
    base = np.array([196, 160, 118], dtype=np.float32)
    return (base[None, None, :] * v[..., None]).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    Image.fromarray(cardboard()).save(OUT / "cardboard.png")
    Image.fromarray(wood()).save(OUT / "wood.png")
    print("wrote", *sorted(p.name for p in OUT.iterdir()))
