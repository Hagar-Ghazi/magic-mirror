import cv2
import numpy as np

from spells.base import Spell

STYLES = ["Cartoon", "Pencil sketch", "Vintage", "Emboss", "Neon edges"]


def cartoon(frame, p):
    color = frame
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 75, 75)
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    result = cv2.bitwise_and(color, color, mask=edges)
    return result, [("Bilateral smoothing", color), ("Adaptive threshold edges", edges)]


def pencil(frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return sketch, [("Grayscale", gray), ("Inverted", inverted), ("Blurred inverse", blurred)]


def vintage(frame, p):
    # rows/columns in BGR order
    sepia = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)
    toned = np.clip(cv2.transform(frame.astype(np.float32), sepia), 0, 255)
    h, w = frame.shape[:2]
    vignette = cv2.getGaussianKernel(w, w / 2) @ cv2.getGaussianKernel(h, h / 2).T
    vignette = (vignette.T / vignette.max())[..., None]
    noise = np.random.normal(0, 8, frame.shape)
    result = np.clip(toned * vignette + noise, 0, 255).astype(np.uint8)
    return result, [("Sepia tone", toned.astype(np.uint8)), ("Vignette mask", (vignette[..., 0] * 255).astype(np.uint8))]


def emboss(frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    result = cv2.filter2D(gray, -1, kernel)
    return result, [("Grayscale", gray)]


def neon(frame, p):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, p["canny"][0], p["canny"][1]), None)
    hue = np.full_like(edges, p["hue"])
    colored = cv2.cvtColor(cv2.merge([hue, np.full_like(edges, 255), edges]), cv2.COLOR_HSV2BGR)
    glow = cv2.GaussianBlur(colored, (15, 15), 0)
    dark = (frame * 0.15).astype(np.uint8)
    result = cv2.add(cv2.add(dark, glow), colored)
    return result, [("Canny edges", edges), ("Glow", glow)]


FILTERS = dict(zip(STYLES, [cartoon, pencil, vintage, emboss, neon]))


class ArtFilters(Spell):
    name = "Art filters"
    icon = "🎨"
    tagline = "Turn yourself into a cartoon, a pencil sketch or a neon sign."
    explain = """
Each style is a short chain of classic operations:

- **Cartoon** - bilateral filter flattens colors but keeps edges; adaptive threshold draws ink lines; `bitwise_and` combines them (Lab04, Lab05 thresholding).
- **Pencil sketch** - divide the gray image by its blurred negative ("color dodge") (Lab03 arithmetic, Lab04 blur).
- **Vintage** - a 3x3 sepia color matrix, a Gaussian vignette and a bit of film grain.
- **Emboss** - a custom convolution kernel with `filter2D` (Lab04 kernels).
- **Neon edges** - Canny edges colored in HSV and blurred to glow (Lab04 edge detection).

Brightness and contrast are `alpha * img + beta` with clipping (Lab03).
"""

    def sidebar(self, ui, mode):
        p = {"style": ui.selectbox("Style", STYLES, key="art_style")}
        if p["style"] == "Neon edges":
            p["canny"] = ui.slider("Canny thresholds", 0, 300, (60, 150), key="art_canny")
            p["hue"] = ui.slider("Neon hue", 0, 179, 150, key="art_hue")
        p["contrast"] = ui.slider("Contrast", 0.5, 2.0, 1.0, step=0.1, key="art_contrast")
        p["brightness"] = ui.slider("Brightness", -80, 80, 0, key="art_brightness")
        p["fast"] = ui.toggle("Fast mode (half resolution)", mode == "live", key="art_fast")
        return p

    def process(self, frame, p, state):
        adjusted = cv2.convertScaleAbs(frame, alpha=p["contrast"], beta=p["brightness"])
        work = adjusted
        if p["fast"]:
            work = cv2.resize(adjusted, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result, inner = FILTERS[p["style"]](work, p)
        if p["fast"]:
            result = cv2.resize(result, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result, [("Original", frame), ("Brightness / contrast", adjusted)] + inner
