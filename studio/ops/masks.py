import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, blend_mask, fit_to, gray, hex_to_bgr, put_label, shape_mask

SHAPE_PARAMS = [
    P("select", "shape", "Shape", "Ellipse", options=["Ellipse", "Circle", "Rectangle"]),
    P("slider", "cx", "Center X (%)", 50, 0, 100, 1),
    P("slider", "cy", "Center Y (%)", 50, 0, 100, 1),
    P("slider", "w", "Width (%)", 50, 1, 100, 1),
    P("slider", "h", "Height (%)", 55, 1, 100, 1),
]


@tool("Masks and blending", "Blend with another image", [
    P("image", "other", "Second image"),
    P("select", "mode", "Mode", "Weighted (addWeighted)", options=[
        "Weighted (addWeighted)", "Add (cv2.add, saturates)", "Add (NumPy, wraps around)", "Subtract",
        "Difference", "Multiply", "Screen", "Lighten (max)", "Darken (min)"]),
    P("slider", "alpha", "Weight of this image", 0.5, 0.0, 1.0, 0.05, when=lambda v: v["mode"].startswith("Weighted")),
], explain="""
**Image arithmetic** works pixel by pixel on two images of the same size (the second one is resized to match).
Note the difference between `cv2.add` (values above 255 are clipped to 255) and NumPy `+` on `uint8`
(values **wrap around**: 250 + 10 = 4), which creates strange colors. *Weighted* is a cross-fade
`alpha * a + (1 - alpha) * b`; multiply and screen are the photo-editing blend modes.
""")
def blend(img, other, mode, alpha=0.5):
    if other is None:
        return put_label(img.copy(), "Upload a second image in the sidebar")
    b = fit_to(other, img.shape)
    if mode.startswith("Weighted"):
        return cv2.addWeighted(img, alpha, b, 1 - alpha, 0)
    if mode.startswith("Add (cv2"):
        return cv2.add(img, b)
    if mode.startswith("Add (NumPy"):
        return img + b
    if mode == "Subtract":
        return cv2.subtract(img, b)
    if mode == "Difference":
        return cv2.absdiff(img, b)
    if mode == "Multiply":
        return cv2.multiply(img, b, scale=1 / 255)
    if mode == "Screen":
        return 255 - cv2.multiply(255 - img, 255 - b, scale=1 / 255)
    return np.maximum(img, b) if mode.startswith("Lighten") else np.minimum(img, b)


@tool("Masks and blending", "Shape mask (focus area)", [
    *SHAPE_PARAMS,
    P("slider", "feather", "Soft edge", 15, 0, 100, 1),
    P("select", "outside", "Outside the shape", "Blur", options=["Blur", "Grayscale", "Darken", "Black", "Solid color", "Pixelate"]),
    P("color", "color", "Color", "#FFFFFF", when=lambda v: v["outside"] == "Solid color"),
    P("toggle", "invert", "Apply inside instead", False),
], explain="""
A **mask** is a grayscale image that decides, per pixel, which of two images to show. A shape is drawn in white on
black (`cv2.ellipse` / `cv2.rectangle`), optionally blurred for a soft edge, and used to mix the original with a
processed copy. This is how a "portrait focus" or spotlight effect is made.
""")
def focus_mask(img, shape, cx, cy, w, h, feather, outside, invert, color="#FFFFFF"):
    mask = shape_mask(img.shape[:2], shape, cx, cy, w, h, feather / 100 * min(img.shape[:2]) * 0.1)
    if outside == "Blur":
        other = cv2.GaussianBlur(img, (0, 0), max(3, img.shape[1] / 40))
    elif outside == "Grayscale":
        other = bgr(gray(img))
    elif outside == "Darken":
        other = (img * 0.35).astype(np.uint8)
    elif outside == "Black":
        other = np.zeros_like(img)
    elif outside == "Pixelate":
        H, W = img.shape[:2]
        other = cv2.resize(cv2.resize(img, (max(1, W // 20), max(1, H // 20))), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        other = np.full_like(img, hex_to_bgr(color))
    if invert:
        mask = 255 - mask
    return blend_mask(img, other, mask)


@tool("Masks and blending", "Bitwise operations", [
    P("select", "op", "Operation", "AND", options=["AND", "OR", "XOR", "NOT"]),
    P("select", "with_", "Second operand", "Shape mask", options=["Shape mask", "Second image"], when=lambda v: v["op"] != "NOT"),
    P("image", "other", "Second image", when=lambda v: v["op"] != "NOT" and v["with_"] == "Second image"),
    *[P(**{**p.__dict__, "when": (lambda v: v["op"] != "NOT" and v["with_"] == "Shape mask")}) for p in SHAPE_PARAMS],
], explain="""
Bitwise operators work on the **bits** of every pixel value. With a white (255 = all ones) / black (0) mask:
**AND** keeps the image only where the mask is white, **OR** forces those pixels to white, **XOR** inverts them,
and **NOT** inverts every pixel. `cv2.bitwise_and(img, img, mask=m)` is the standard way to cut out a region.
""")
def bitwise(img, op, with_="Shape mask", other=None, shape="Ellipse", cx=50, cy=50, w=60, h=70):
    if op == "NOT":
        return cv2.bitwise_not(img)
    if with_ == "Second image":
        if other is None:
            return put_label(img.copy(), "Upload a second image in the sidebar")
        b = fit_to(other, img.shape)
    else:
        b = bgr(shape_mask(img.shape[:2], shape, cx, cy, w, h))
    fn = {"AND": cv2.bitwise_and, "OR": cv2.bitwise_or, "XOR": cv2.bitwise_xor}[op]
    return fn(img, b)


@tool("Masks and blending", "Chroma key (replace a color)", [
    P("color", "key", "Color to remove", "#00B140"),
    P("slider", "tolerance", "Hue tolerance", 15, 1, 60, 1),
    P("slider", "min_sat", "Minimum saturation", 60, 0, 255, 1),
    P("select", "replace", "Replace with", "Second image", options=["Second image", "Solid color", "Blurred original", "Transparent checkerboard"]),
    P("image", "other", "Background image", when=lambda v: v["replace"] == "Second image"),
    P("color", "color", "Background color", "#1E293B", when=lambda v: v["replace"] == "Solid color"),
    P("slider", "smooth", "Edge smoothing", 3, 0, 15, 1),
], explain="""
The green-screen effect. The key color is converted to HSV and `cv2.inRange` marks every pixel with a similar hue
(and enough saturation). The mask is cleaned with morphology and softened, then used to mix the photo with a new
background.
""")
def chroma(img, key, tolerance, min_sat, replace, smooth, other=None, color="#1E293B"):
    key_hue = int(cv2.cvtColor(np.uint8([[hex_to_bgr(key)]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = key_hue - tolerance, key_hue + tolerance
    mask = cv2.inRange(hsv, (max(lo, 0), min_sat, 30), (min(hi, 179), 255, 255))
    if lo < 0:
        mask |= cv2.inRange(hsv, (180 + lo, min_sat, 30), (179, 255, 255))
    if hi > 179:
        mask |= cv2.inRange(hsv, (0, min_sat, 30), (hi - 180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if smooth:
        mask = cv2.GaussianBlur(mask, (smooth * 2 + 1, smooth * 2 + 1), 0)
    if replace == "Second image":
        if other is None:
            return put_label(img.copy(), "Upload a background image in the sidebar")
        bg = fit_to(other, img.shape)
    elif replace == "Solid color":
        bg = np.full_like(img, hex_to_bgr(color))
    elif replace == "Blurred original":
        bg = cv2.GaussianBlur(img, (0, 0), 15)
    else:
        s = max(8, img.shape[1] // 40)
        ys, xs = np.indices(img.shape[:2])
        bg = bgr(np.where(((xs // s) + (ys // s)) % 2 == 0, 200, 255).astype(np.uint8))
    return blend_mask(bg, img, mask)
