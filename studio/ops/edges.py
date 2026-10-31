import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, gray, hex_to_bgr, odd

DISPLAY = [
    P("select", "display", "Display", "Edges only", options=["Edges only", "Inverted (black on white)", "Overlay on image"]),
    P("color", "color", "Edge color", "#00FF88", when=lambda v: v["display"] == "Overlay on image"),
]


def _show(img, edges, display, color="#00FF88"):
    edges = cv2.convertScaleAbs(edges)
    if display == "Edges only":
        return bgr(edges)
    if display == "Inverted (black on white)":
        return bgr(255 - edges)
    out = img.copy()
    out[edges > 40] = hex_to_bgr(color)
    return out


def _blurred_gray(img, k):
    g = gray(img)
    return cv2.GaussianBlur(g, (odd(k), odd(k)), 0) if k > 1 else g


@tool("Edges", "Canny", [
    P("range", "thresholds", "Low / high threshold", (50, 150), 0, 500, 5),
    P("slider", "blur", "Blur first (kernel)", 5, 1, 15, 2),
    P("toggle", "l2", "Accurate gradient (L2)", False),
    *DISPLAY,
], explain="""
`cv2.Canny` is a multi-stage detector: gradient with Sobel, **non-maximum suppression** to thin edges to 1 px, then
**hysteresis**: pixels above the high threshold are edges, pixels between the two thresholds are kept only if they
touch a strong edge. Blurring first removes noise that would create fake edges.
""")
def canny(img, thresholds, blur, l2, display, color="#00FF88"):
    edges = cv2.Canny(_blurred_gray(img, blur), thresholds[0], thresholds[1], L2gradient=l2)
    return _show(img, edges, display, color)


@tool("Edges", "Sobel", [
    P("select", "direction", "Direction", "Magnitude (both)", options=["X (vertical edges)", "Y (horizontal edges)", "Magnitude (both)"]),
    P("select", "ksize", "Kernel size", 3, options=[1, 3, 5, 7]),
    P("slider", "blur", "Blur first (kernel)", 3, 1, 15, 2),
    *DISPLAY,
], explain="""
`cv2.Sobel` computes the image **derivative**: how fast brightness changes. The X derivative responds to vertical
edges, the Y derivative to horizontal ones, and the magnitude `sqrt(gx^2 + gy^2)` to both. Results are computed in
`CV_64F` so negative slopes are not lost, then converted back with `convertScaleAbs`.
""")
def sobel(img, direction, ksize, blur, display, color="#00FF88"):
    g = _blurred_gray(img, blur)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=int(ksize))
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=int(ksize))
    edges = {"X (vertical edges)": np.abs(gx), "Y (horizontal edges)": np.abs(gy)}.get(direction, np.hypot(gx, gy))
    return _show(img, edges, display, color)


@tool("Edges", "Laplacian", [
    P("select", "ksize", "Kernel size", 3, options=[1, 3, 5, 7]),
    P("slider", "blur", "Blur first (kernel)", 3, 1, 15, 2),
    *DISPLAY,
], explain="""
The **Laplacian** is the second derivative: it is zero on flat areas and on smooth slopes, and large where the slope
itself changes - at edges, in all directions at once. It is very sensitive to noise, so blur first.
""")
def laplacian(img, ksize, blur, display, color="#00FF88"):
    edges = np.abs(cv2.Laplacian(_blurred_gray(img, blur), cv2.CV_64F, ksize=int(ksize)))
    return _show(img, edges, display, color)


@tool("Edges", "Scharr", [
    P("select", "direction", "Direction", "Magnitude (both)", options=["X (vertical edges)", "Y (horizontal edges)", "Magnitude (both)"]),
    *DISPLAY,
], explain="""
`cv2.Scharr` is a 3x3 derivative like Sobel but with weights (3, 10, 3) that make it more accurate for diagonal
edges and fine detail.
""")
def scharr(img, direction, display, color="#00FF88"):
    g = gray(img)
    gx, gy = cv2.Scharr(g, cv2.CV_64F, 1, 0), cv2.Scharr(g, cv2.CV_64F, 0, 1)
    edges = {"X (vertical edges)": np.abs(gx), "Y (horizontal edges)": np.abs(gy)}.get(direction, np.hypot(gx, gy))
    return _show(img, edges / 4, display, color)


@tool("Edges", "Prewitt", [
    P("select", "direction", "Direction", "Magnitude (both)", options=["X (vertical edges)", "Y (horizontal edges)", "Magnitude (both)"]),
    *DISPLAY,
], explain="""
Prewitt is the simplest derivative kernel, `[-1 0 1]` repeated on three rows, applied with `cv2.filter2D`.
Comparing it with Sobel shows what the extra center weight does.
""")
def prewitt(img, direction, display, color="#00FF88"):
    g = gray(img).astype(np.float32)
    kx = np.array([[-1, 0, 1]] * 3, np.float32)
    gx, gy = cv2.filter2D(g, -1, kx), cv2.filter2D(g, -1, kx.T)
    edges = {"X (vertical edges)": np.abs(gx), "Y (horizontal edges)": np.abs(gy)}.get(direction, np.hypot(gx, gy))
    return _show(img, edges, display, color)
