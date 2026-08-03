"""
Shared detection/warp logic for finding a painting on a plain wall and
straightening it.

Two detection strategies are provided:

- find_corners_colormask(): segments by color distance from the wall
  (sampled from the image corners), which is far more robust than plain
  edge detection when the frame/border is thin, inconsistent, or partially
  missing in places.

- warp_to_rectangle(): given 4 corners (in any order), perspective-warps
  them to an upright rectangle.

IMPORTANT LIMITATION: if part of the true canvas edge is (close to) the
same color as the wall -- e.g. a white canvas margin against a beige wall
with no border paint in that spot -- there is *no visual signal* left to
find that edge automatically. No color- or edge-based method can solve
this reliably, because the pixels themselves don't encode where the
boundary is. That's what the manual-correction UI in app.py is for:
auto-detection gives a good starting guess, a human corrects the last mile.
"""

import cv2
import numpy as np


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def find_corners_colormask(img, corner_patch=60, dist_thresh=6.0):
    """
    Detect the painting by segmenting pixels that differ from the wall
    color (sampled from the 4 corners of the photo), rather than relying
    on edges. Returns ordered corners [tl, tr, br, bl] or None.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    p = corner_patch
    corner_samples = [
        lab[0:p, 0:p], lab[0:p, w - p:w],
        lab[h - p:h, 0:p], lab[h - p:h, w - p:w],
    ]
    wall_samples = np.concatenate([c.reshape(-1, 3) for c in corner_samples], axis=0)
    wall_mean = wall_samples.mean(axis=0)
    wall_std = wall_samples.std(axis=0) + 1e-6

    diff = (lab - wall_mean) / wall_std
    dist = np.sqrt((diff ** 2).sum(axis=2))
    mask = (dist > dist_thresh).astype(np.uint8) * 255

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    img_area = h * w
    if cv2.contourArea(c) < 0.05 * img_area:
        return None

    # Convex hull: if the border is missing/low-contrast along one edge,
    # the mask can dip inward there (a concave notch). The hull bridges
    # straight across that notch, which is correct for a physically
    # rectangular canvas -- but it CANNOT fix the opposite problem of the
    # mask bulging outward into the wall. Hence: good default, not a fix
    # for every case.
    hull = cv2.convexHull(c)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return order_points(box)


def warp_to_rectangle(img, corners, padding=0):
    """Perspective-warp the quad defined by corners to an upright rectangle."""
    corners = np.array(corners, dtype="float32")
    (tl, tr, br, bl) = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    if padding > 0:
        warped = warped[padding:-padding, padding:-padding]
    return warped
