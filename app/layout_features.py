"""
Page-layout features for finding ad areas on a newspaper page.

The page (resized to WORK_LONG_SIDE) is divided into square cells. Each cell is
described by visual measures computed at three scales (the cell itself, a small
neighbourhood and a wide neighbourhood), so the model can tell "fine body text
of a news column" from "display area of an ad" (colour, pictures, big letters,
solid panels, boxes).

Used by app/page_scan.py at scan time and tools/train_layout.py for training.
"""
from __future__ import annotations

import numpy as np

CELL = 24                 # cell size in work-image pixels
SCALES = (1, 3, 9)        # neighbourhood sizes in cells
MAPS = ("sat", "val", "val_std", "dark", "ink", "edges", "thick", "rules", "white")


def pixel_maps(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel measures (float32 in 0..1) for an RGB uint8 image."""
    import cv2

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    H, W = gray.shape
    g = gray.astype(np.float32) / 255
    sat = (hsv[:, :, 1].astype(np.float32) / 255) * (hsv[:, :, 2] > 40)
    ink = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12) > 0).astype(np.uint8)
    edges = (cv2.Canny(gray, 60, 160) > 0).astype(np.float32)
    # "thick" = ink that survives erosion: big letters, solid shapes, photos (body text strokes vanish)
    k = max(3, W // 500)
    thick = cv2.erode(ink, np.ones((k, k), np.uint8)).astype(np.float32)
    # ruled lines
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 30, 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 30, 15)))
    rules = ((cv2.morphologyEx(ink * 255, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(ink * 255, cv2.MORPH_OPEN, vk)) > 0).astype(np.float32)
    # local brightness variation (photos / gradients vs flat paper)
    mu = cv2.blur(g, (9, 9)); mu2 = cv2.blur(g * g, (9, 9))
    val_std = np.sqrt(np.clip(mu2 - mu * mu, 0, None)) * 3
    white = (g > 0.85).astype(np.float32)
    return {"sat": sat, "val": g, "val_std": np.clip(val_std, 0, 1), "dark": (g < 0.3).astype(np.float32),
            "ink": ink.astype(np.float32), "edges": edges, "thick": thick, "rules": rules, "white": white}


def cell_features(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (features[rows*cols, F], (rows, cols)) for an RGB work image."""
    import cv2

    maps = pixel_maps(arr)
    H, W = arr.shape[:2]
    rows, cols = H // CELL, W // CELL
    per_cell = []
    for name in MAPS:
        m = maps[name][: rows * CELL, : cols * CELL]
        per_cell.append(m.reshape(rows, CELL, cols, CELL).mean(axis=(1, 3)))
    base = np.stack(per_cell, axis=-1)                      # rows x cols x M
    feats = []
    for s in SCALES:
        feats.append(base if s == 1 else cv2.blur(base, (s, s), borderType=cv2.BORDER_REFLECT))
    yy, xx = np.mgrid[0:rows, 0:cols]
    pos = np.stack([yy / max(1, rows - 1), xx / max(1, cols - 1),
                    np.minimum(np.minimum(yy, rows - 1 - yy), np.minimum(xx, cols - 1 - xx)) / max(rows, cols)], axis=-1)
    F = np.concatenate(feats + [pos], axis=-1)
    # a few interactions the linear model can't form itself
    thick9, sat9, ink9, edges9 = F[..., 2 * len(MAPS) + MAPS.index("thick")], F[..., 2 * len(MAPS) + MAPS.index("sat")], \
        F[..., 2 * len(MAPS) + MAPS.index("ink")], F[..., 2 * len(MAPS) + MAPS.index("edges")]
    extra = np.stack([thick9 * sat9, ink9 * (1 - thick9), edges9 * (1 - sat9), np.sqrt(sat9), np.sqrt(thick9)], axis=-1)
    F = np.concatenate([F, extra], axis=-1)
    # the same measures relative to this page (robust to paper colour, print style and photo quality)
    flat = F.reshape(rows * cols, -1)
    n_rel = flat.shape[1] - 3 - extra.shape[-1]          # skip position columns
    rel = (flat[:, :n_rel] - np.median(flat[:, :n_rel], axis=0)) / (flat[:, :n_rel].std(axis=0) + 1e-3)
    flat = np.concatenate([flat, np.clip(rel, -6, 6)], axis=1)
    return flat.astype(np.float32), (rows, cols)
