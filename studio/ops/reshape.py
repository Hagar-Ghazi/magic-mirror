import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import detect_faces, put_label


def _radial_warp(img, cx, cy, radius, strength=0.0, twist=0.0):
    """Bulge/pinch (strength) and swirl (twist, radians) inside a circle, smooth at its edge.

    Only the bounding box of the circle is remapped, which keeps it fast on big images.
    """
    h, w = img.shape[:2]
    radius = max(2, int(radius))
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    d = np.sqrt(dx * dx + dy * dy) / radius
    falloff = np.clip(1 - d, 0, 1) ** 2
    scale = 1 - strength * falloff  # < 1 samples nearer the center = magnify
    theta = twist * falloff
    cos, sin = np.cos(theta), np.sin(theta)
    map_x = (cx + (dx * cos - dy * sin) * scale).astype(np.float32)
    map_y = (cy + (dx * sin + dy * cos) * scale).astype(np.float32)
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


@tool("Reshape", "Face size (enlarge or shrink)", [
    P("select", "target", "Target", "Largest face", options=["Largest face", "All faces", "Manual point"]),
    P("slider", "strength", "Size change", 0.4, -0.9, 0.9, 0.05, help="Positive enlarges, negative shrinks"),
    P("slider", "reach", "Area around the face", 1.0, 0.5, 2.0, 0.05, when=lambda v: v["target"] != "Manual point"),
    P("slider", "x", "Center X (%)", 50, 0, 100, 1, when=lambda v: v["target"] == "Manual point"),
    P("slider", "y", "Center Y (%)", 40, 0, 100, 1, when=lambda v: v["target"] == "Manual point"),
    P("slider", "radius", "Radius (% of width)", 20, 2, 80, 1, when=lambda v: v["target"] == "Manual point"),
], explain="""
Faces are found with a **Haar cascade** (on a histogram-equalized gray image). Around each face a smooth
**bulge or pinch** is applied with `cv2.remap`: for every output pixel we compute *where to read from*.
Reading from closer to the center magnifies (bigger face); reading from farther shrinks it. The effect fades to
zero at the circle edge (`(1 - d)^2`), so there is no visible seam.
""")
def face_size(img, target, strength, reach=1.0, x=50, y=40, radius=20):
    h, w = img.shape[:2]
    if target == "Manual point":
        return _radial_warp(img, x / 100 * w, y / 100 * h, radius / 100 * w, strength)
    faces = detect_faces(img)
    if not faces:
        return put_label(img.copy(), "No face found - try Manual point")
    if target == "Largest face":
        faces = [max(faces, key=lambda f: f[2] * f[3])]
    out = img
    for fx, fy, fw, fh in faces:
        out = _radial_warp(out, fx + fw / 2, fy + fh / 2, 0.75 * max(fw, fh) * reach, strength)
    return out


@tool("Reshape", "Bulge or pinch", [
    P("slider", "x", "Center X (%)", 50, 0, 100, 1),
    P("slider", "y", "Center Y (%)", 50, 0, 100, 1),
    P("slider", "radius", "Radius (% of width)", 30, 2, 100, 1),
    P("slider", "strength", "Strength", 0.75, -0.9, 0.9, 0.05, help="Positive bulges, negative pinches"),
], explain="""
The same smooth radial warp used for faces, at any point: a **fish-eye bulge** (positive) or a **pinch**
(negative). Implemented with `cv2.remap` and maps computed with NumPy.
""")
def bulge(img, x, y, radius, strength):
    h, w = img.shape[:2]
    return _radial_warp(img, x / 100 * w, y / 100 * h, radius / 100 * w, strength)


@tool("Reshape", "Swirl", [
    P("slider", "x", "Center X (%)", 50, 0, 100, 1),
    P("slider", "y", "Center Y (%)", 50, 0, 100, 1),
    P("slider", "radius", "Radius (% of width)", 35, 2, 100, 1),
    P("slider", "angle", "Twist (degrees)", 180, -720, 720, 10),
], explain="""
Each pixel inside the circle is rotated around the center by an angle that is largest in the middle and fades to
zero at the edge, then sampled with `cv2.remap`.
""")
def swirl(img, x, y, radius, angle):
    h, w = img.shape[:2]
    return _radial_warp(img, x / 100 * w, y / 100 * h, radius / 100 * w, 0.0, np.deg2rad(angle))


@tool("Reshape", "Slim or widen", [
    P("slider", "cx", "Body center X (%)", 50, 0, 100, 1),
    P("slider", "half_width", "Body half-width (%)", 32, 5, 50, 1),
    P("slider", "strength", "Slim (+) or widen (-)", 0.5, -0.7, 0.7, 0.05),
    P("range", "rows", "Apply to rows (%)", (0, 100), 0, 100, 1),
], explain="""
A horizontal-only warp: inside the band around the body center, pixels are read from farther away (slimmer) or
closer (wider) with a smooth falloff, so the background just outside the band is untouched. Limiting the
rows lets you slim only the waist or only the legs; the edges of the row band are feathered.
""")
def slim(img, cx, half_width, strength, rows):
    h, w = img.shape[:2]
    c, hw = cx / 100 * w, max(2, half_width / 100 * w)
    xs = np.arange(w, dtype=np.float32)
    dx = xs - c
    d = np.abs(dx) / hw
    scale = 1 + strength * np.clip(1 - d, 0, 1) ** 2
    map_x = np.tile(c + dx * scale, (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if rows == (0, 100):
        return warped
    weight = np.zeros(h, np.float32)
    weight[int(rows[0] / 100 * h):int(rows[1] / 100 * h)] = 1
    sigma = max(1.0, h * 0.02)
    kernel = cv2.getGaussianKernel(int(sigma * 6) | 1, sigma).ravel()
    weight = np.convolve(weight, kernel, mode="same")
    m = weight[:, None, None]
    return (warped * m + img * (1 - m)).astype(np.uint8)


@tool("Reshape", "Height stretch (taller, longer legs)", [
    P("range", "band", "Stretch rows (%)", (55, 100), 0, 100, 1, help="For longer legs choose the lower part of the body"),
    P("slider", "factor", "Stretch factor", 1.3, 0.5, 1.8, 0.01),
], explain="""
The image is cut into three horizontal strips. Only the middle strip is resized vertically with `cv2.resize`,
then the strips are stacked again with `np.vstack`. Stretching the legs region makes a person look taller;
a factor below 1 makes them shorter. The image height changes, the width does not.
""")
def height_stretch(img, band, factor):
    h = img.shape[0]
    y1, y2 = int(band[0] / 100 * h), int(band[1] / 100 * h)
    if y2 - y1 < 2:
        return img
    mid = img[y1:y2]
    mid = cv2.resize(mid, (img.shape[1], max(1, int(mid.shape[0] * factor))), interpolation=cv2.INTER_CUBIC)
    return np.vstack([img[:y1], mid, img[y2:]])
