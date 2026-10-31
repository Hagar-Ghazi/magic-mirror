import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import hex_to_bgr

INTERP = {
    "Nearest": cv2.INTER_NEAREST,
    "Linear": cv2.INTER_LINEAR,
    "Area": cv2.INTER_AREA,
    "Cubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
}
BORDERS = {
    "Constant color": cv2.BORDER_CONSTANT,
    "Reflect": cv2.BORDER_REFLECT,
    "Reflect 101": cv2.BORDER_REFLECT_101,
    "Replicate": cv2.BORDER_REPLICATE,
    "Wrap": cv2.BORDER_WRAP,
}
BORDER_PARAMS = [
    P("select", "border", "Empty area", "Constant color", options=list(BORDERS)),
    P("color", "fill", "Fill color", "#000000", when=lambda v: v["border"] == "Constant color"),
]


def _warp_affine(img, m, size, border, fill):
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderMode=BORDERS[border], borderValue=hex_to_bgr(fill))


@tool("Geometry", "Resize", [
    P("select", "mode", "Mode", "Scale by percent", options=["Scale by percent", "Exact size"]),
    P("slider", "scale", "Scale (%)", 50, 5, 400, 5, when=lambda v: v["mode"] == "Scale by percent"),
    P("slider", "width", "Width (px)", 800, 16, 4000, 8, when=lambda v: v["mode"] == "Exact size"),
    P("slider", "height", "Height (px)", 600, 16, 4000, 8, when=lambda v: v["mode"] == "Exact size"),
    P("toggle", "keep_aspect", "Keep aspect ratio (use width)", True, when=lambda v: v["mode"] == "Exact size"),
    P("select", "interp", "Interpolation", "Area", options=list(INTERP)),
], explain="""
`cv2.resize` changes the number of pixels. **Interpolation** decides how new pixel values are computed:
*Nearest* copies the closest pixel (blocky), *Linear*/*Cubic*/*Lanczos* average neighbours (smooth), and
*Area* is best for shrinking. Keeping the **aspect ratio** (`width / height`) avoids stretching.
""")
def resize(img, mode, interp, scale=50, width=800, height=600, keep_aspect=True):
    h, w = img.shape[:2]
    if mode == "Scale by percent":
        size = (max(1, int(w * scale / 100)), max(1, int(h * scale / 100)))
    elif keep_aspect:
        size = (int(width), max(1, int(round(width * h / w))))
    else:
        size = (int(width), int(height))
    return cv2.resize(img, size, interpolation=INTERP[interp])


@tool("Geometry", "Crop", [
    P("range", "x", "Horizontal (%)", (10, 90), 0, 100, 1),
    P("range", "y", "Vertical (%)", (10, 90), 0, 100, 1),
], explain="""
Cropping is just **NumPy slicing**: `img[y1:y2, x1:x2]`. The result is a *view* into the same memory
(Region Of Interest), so tools copy it before changing it.
""")
def crop(img, x, y):
    h, w = img.shape[:2]
    x1, x2 = int(x[0] / 100 * w), max(int(x[1] / 100 * w), int(x[0] / 100 * w) + 1)
    y1, y2 = int(y[0] / 100 * h), max(int(y[1] / 100 * h), int(y[0] / 100 * h) + 1)
    return img[y1:y2, x1:x2].copy()


@tool("Geometry", "Crop to aspect ratio", [
    P("select", "ratio", "Aspect ratio", "1:1", options=["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "4:5"]),
    P("slider", "cx", "Center X (%)", 50, 0, 100, 1),
    P("slider", "cy", "Center Y (%)", 50, 0, 100, 1),
], explain="""
Finds the largest window with the chosen **aspect ratio** that fits in the image, centers it on the chosen point
(clamped to the borders) and slices it out. Useful for profile pictures (1:1) or stories (9:16).
""")
def crop_ratio(img, ratio, cx, cy):
    h, w = img.shape[:2]
    a, b = (int(v) for v in ratio.split(":"))
    target = a / b
    cw, ch = (w, int(w / target)) if w / h < target else (int(h * target), h)
    x1 = int(np.clip(cx / 100 * w - cw / 2, 0, w - cw))
    y1 = int(np.clip(cy / 100 * h - ch / 2, 0, h - ch))
    return img[y1:y1 + ch, x1:x1 + cw].copy()


@tool("Geometry", "Rotate", [
    P("slider", "angle", "Angle (degrees, counter-clockwise)", 15, -180, 180, 1),
    P("slider", "scale", "Scale", 1.0, 0.2, 3.0, 0.05),
    P("toggle", "expand", "Grow canvas to fit", True),
    *BORDER_PARAMS,
], explain="""
`cv2.getRotationMatrix2D(center, angle, scale)` builds a 2x3 **affine matrix**; `cv2.warpAffine` moves every pixel
with it. Rotating cuts off the corners unless the canvas grows: the new size is
`w*|cos| + h*|sin|` by `w*|sin| + h*|cos|`, and the matrix is shifted to the new center.
""")
def rotate(img, angle, scale, expand, border, fill="#000000"):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    size = (w, h)
    if expand:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        size = (nw, nh)
    return _warp_affine(img, m, size, border, fill)


@tool("Geometry", "Rotate 90 degrees", [
    P("select", "turn", "Turn", "90 clockwise", options=["90 clockwise", "180", "90 counter-clockwise"]),
], explain="""
Right-angle turns need no interpolation, so `cv2.rotate` just rearranges pixels: no blur and no lost corners.
""")
def rotate90(img, turn):
    code = {"90 clockwise": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "90 counter-clockwise": cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, code[turn])


@tool("Geometry", "Flip", [
    P("select", "direction", "Direction", "Horizontal (mirror)", options=["Horizontal (mirror)", "Vertical", "Both"]),
], explain="""
`cv2.flip(img, code)`: code 1 mirrors left-right, 0 flips upside-down, -1 does both (same as a 180 degree rotation).
""")
def flip(img, direction):
    return cv2.flip(img, {"Horizontal (mirror)": 1, "Vertical": 0, "Both": -1}[direction])


@tool("Geometry", "Translate (shift)", [
    P("slider", "dx", "Shift X (% of width)", 10, -100, 100, 1),
    P("slider", "dy", "Shift Y (% of height)", 10, -100, 100, 1),
    *BORDER_PARAMS,
], explain="""
Translation uses the affine matrix `[[1, 0, tx], [0, 1, ty]]` with `cv2.warpAffine`. Pixels that come from outside
the image are filled according to the **border mode** (constant color, reflect, replicate or wrap).
""")
def translate(img, dx, dy, border, fill="#000000"):
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx / 100 * w], [0, 1, dy / 100 * h]])
    return _warp_affine(img, m, (w, h), border, fill)


@tool("Geometry", "Shear (affine)", [
    P("slider", "shear_x", "Shear X", 0.2, -1.0, 1.0, 0.05),
    P("slider", "shear_y", "Shear Y", 0.0, -1.0, 1.0, 0.05),
    *BORDER_PARAMS,
], explain="""
An **affine transform** keeps straight lines straight and parallel lines parallel. It is fully defined by where
three points go: `cv2.getAffineTransform(src3, dst3)` computes the matrix. Here the top-left, top-right and
bottom-left corners are moved to slant the image, and the canvas grows to fit.
""")
def shear(img, shear_x, shear_y, border, fill="#000000"):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h]])
    dst = np.float32([[0, 0], [w, w * shear_y], [h * shear_x, h]])
    pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    m = cv2.getAffineTransform(src, dst)
    moved = cv2.transform(pts[None], m)[0]
    min_xy = moved.min(axis=0)
    m[:, 2] -= min_xy
    size = tuple(int(v) for v in np.ceil(moved.max(axis=0) - min_xy))
    return _warp_affine(img, m, size, border, fill)


@tool("Geometry", "Perspective (tilt)", [
    P("slider", "top", "Squeeze top edge (%)", 15, -40, 40, 1),
    P("slider", "bottom", "Squeeze bottom edge (%)", 0, -40, 40, 1),
    P("slider", "left", "Squeeze left edge (%)", 0, -40, 40, 1),
    P("slider", "right", "Squeeze right edge (%)", 0, -40, 40, 1),
    *BORDER_PARAMS,
], explain="""
A **perspective transform** maps any four points to any four points (lines stay straight but parallel lines may
meet, like railway tracks). `cv2.getPerspectiveTransform` builds the 3x3 matrix from the four corners and
`cv2.warpPerspective` applies it. Squeezing the top edge makes the image lean backwards.
""")
def perspective(img, top, bottom, left, right, border, fill="#000000"):
    h, w = img.shape[:2]
    t, b, l, r = (v / 100 for v in (top, bottom, left, right))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * t / 2, h * l / 2],
        [w - w * t / 2, h * r / 2],
        [w - w * b / 2, h - h * r / 2],
        [w * b / 2, h - h * l / 2],
    ])
    dst -= dst.min(axis=0)
    size = tuple(int(v) for v in np.ceil(dst.max(axis=0)))
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, size, borderMode=BORDERS[border], borderValue=hex_to_bgr(fill))
