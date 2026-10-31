import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, detect_faces, gray, hex_to_bgr, odd, put_label, shape_mask

COLORMAPS = {
    "Jet": cv2.COLORMAP_JET, "Hot": cv2.COLORMAP_HOT, "Inferno": cv2.COLORMAP_INFERNO, "Magma": cv2.COLORMAP_MAGMA,
    "Viridis": cv2.COLORMAP_VIRIDIS, "Ocean": cv2.COLORMAP_OCEAN, "Rainbow": cv2.COLORMAP_RAINBOW,
    "Bone": cv2.COLORMAP_BONE, "Pink": cv2.COLORMAP_PINK, "Twilight": cv2.COLORMAP_TWILIGHT, "Turbo": cv2.COLORMAP_TURBO,
}


def _sepia(img):
    m = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)  # BGR order
    return np.clip(cv2.transform(img.astype(np.float32), m), 0, 255).astype(np.uint8)


def _vignette_mask(h, w, radius, cx=50, cy=50):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - cx / 100 * w) / w) ** 2 + ((ys - cy / 100 * h) / h) ** 2)
    return np.clip(1 - (d / max(radius, 0.05)) ** 2, 0, 1)


@tool("Artistic", "Cartoon", [
    P("slider", "passes", "Color smoothing passes", 2, 1, 6, 1),
    P("slider", "block", "Line detail (block size)", 9, 3, 31, 2),
    P("slider", "c", "Line strength (C)", 5, 1, 20, 1),
    P("slider", "levels", "Color levels (0 = off)", 0, 0, 12, 1, help="2 or more reduces the colors"),
], explain="""
1. `cv2.bilateralFilter` several times flattens colors but keeps edges.
2. Median blur + `cv2.adaptiveThreshold` on the gray image draws dark ink lines.
3. `cv2.bitwise_and` with the line mask combines them. Optional posterizing reduces the colors further.
""")
def cartoon(img, passes, block, c, levels):
    color = img
    for _ in range(passes):
        color = cv2.bilateralFilter(color, 9, 75, 75)
    if levels >= 2:
        step = 256 // levels
        color = ((color // step) * step + step // 2).astype(np.uint8)
    g = cv2.medianBlur(gray(img), 7)
    lines = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, odd(block), c)
    return cv2.bitwise_and(color, color, mask=lines)


@tool("Artistic", "Pencil sketch", [
    P("select", "style", "Style", "Gray pencil", options=["Gray pencil", "Color pencil"]),
    P("slider", "softness", "Softness", 21, 3, 81, 2),
], explain="""
**Color dodge:** `gray / (255 - blurred_inverse) * 256` (with `cv2.divide`). Where the image is flat the result is
white paper; at edges the blurred negative differs from the original and dark strokes appear. The color version
multiplies the sketch with the original colors.
""")
def pencil(img, style, softness):
    g = gray(img)
    blurred = cv2.GaussianBlur(255 - g, (odd(softness), odd(softness)), 0)
    sketch = cv2.divide(g, 255 - blurred, scale=256)
    if style == "Gray pencil":
        return bgr(sketch)
    return cv2.multiply(img, bgr(sketch), scale=1 / 255)


@tool("Artistic", "Sepia", [
    P("slider", "intensity", "Intensity", 1.0, 0.0, 1.0, 0.05),
], explain="""
A 3x3 **color matrix** with `cv2.transform` mixes the B, G, R channels into warm brown tones (each output channel is
a weighted sum of the inputs). Intensity blends it with the original.
""")
def sepia(img, intensity):
    return cv2.addWeighted(_sepia(img), intensity, img, 1 - intensity, 0)


@tool("Artistic", "Vintage photo", [
    P("slider", "fade", "Fade", 0.25, 0.0, 0.6, 0.05),
    P("slider", "grain", "Film grain", 10, 0, 40, 1),
    P("slider", "vignette", "Vignette", 0.6, 0.0, 1.0, 0.05),
], explain="""
Sepia toning, a lifted black level (fade), random **film grain** (Gaussian noise) and a darker border (vignette mask)
multiplied together.
""")
def vintage(img, fade, grain, vignette):
    out = _sepia(img).astype(np.float32)
    out = out * (1 - fade) + 255 * fade * 0.5
    out += np.random.default_rng(3).normal(0, grain, img.shape) if grain else 0
    mask = _vignette_mask(*img.shape[:2], 0.9)
    out *= (1 - vignette + vignette * mask)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


@tool("Artistic", "Vignette", [
    P("slider", "strength", "Strength", 0.7, 0.0, 1.0, 0.05),
    P("slider", "radius", "Radius", 0.8, 0.2, 1.5, 0.05),
    P("slider", "cx", "Center X (%)", 50, 0, 100, 1),
    P("slider", "cy", "Center Y (%)", 50, 0, 100, 1),
    P("color", "color", "Edge color", "#000000"),
], explain="""
A smooth radial mask (1 in the center, 0 at the edge) blends the photo with a solid color, drawing the eye to the
center. This is **masking with a soft (grayscale) mask** instead of a binary one.
""")
def vignette(img, strength, radius, cx, cy, color):
    mask = _vignette_mask(*img.shape[:2], radius, cx, cy)
    m = (1 - strength + strength * mask)[..., None]
    solid = np.full_like(img, hex_to_bgr(color), dtype=np.float32)
    return (img * m + solid * (1 - m)).astype(np.uint8)


@tool("Artistic", "Emboss", [
    P("select", "direction", "Light from", "Top-left", options=["Top-left", "Top", "Left", "Top-right"]),
    P("slider", "strength", "Strength", 1.0, 0.2, 3.0, 0.1),
    P("toggle", "keep_color", "Keep colors", False),
], explain="""
An emboss kernel subtracts pixels on one side from pixels on the other, so edges facing the light turn bright and
edges facing away turn dark. `cv2.filter2D` applies it with an offset of 128 (mid gray) for flat areas.
""")
def emboss(img, direction, strength, keep_color):
    base = {"Top-left": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], "Top": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "Left": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "Top-right": [[0, -1, -2], [1, 0, -1], [2, 1, 0]]}[direction]
    kernel = np.array(base, np.float32) * strength
    relief = cv2.filter2D(gray(img), -1, kernel, delta=128)
    if keep_color:  # shade the colors with the relief (128 = unchanged)
        return np.clip(img * (relief[..., None] / 128.0), 0, 255).astype(np.uint8)
    return bgr(relief)


@tool("Artistic", "Pixelate", [
    P("slider", "blocks", "Block size (px)", 16, 2, 80, 1),
    P("select", "region", "Region", "Whole image", options=["Whole image", "Faces (privacy)", "Shape"]),
    P("select", "shape", "Shape", "Ellipse", options=["Rectangle", "Ellipse", "Circle"], when=lambda v: v["region"] == "Shape"),
    P("slider", "cx", "Center X (%)", 50, 0, 100, 1, when=lambda v: v["region"] == "Shape"),
    P("slider", "cy", "Center Y (%)", 50, 0, 100, 1, when=lambda v: v["region"] == "Shape"),
    P("slider", "w", "Width (%)", 40, 1, 100, 1, when=lambda v: v["region"] == "Shape"),
    P("slider", "h", "Height (%)", 40, 1, 100, 1, when=lambda v: v["region"] == "Shape"),
], explain="""
Shrink with `cv2.resize` to a few pixels, then enlarge back with **nearest-neighbour** interpolation: every small
pixel becomes a visible block. Restricting it to detected **faces** (Haar cascade) or a shape mask is a classic
privacy filter.
""")
def pixelate(img, blocks, region, shape="Ellipse", cx=50, cy=50, w=40, h=40):
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(1, W // blocks), max(1, H // blocks)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    if region == "Whole image":
        return pix
    if region == "Shape":
        mask = shape_mask((H, W), shape, cx, cy, w, h)
        return np.where(mask[..., None] > 0, pix, img)
    faces = detect_faces(img)
    if not faces:
        return put_label(img.copy(), "No face found")
    out = img.copy()
    for x, y, fw, fh in faces:
        roi = out[y:y + fh, x:x + fw]
        tiny = cv2.resize(roi, (max(1, fw // blocks), max(1, fh // blocks)))
        out[y:y + fh, x:x + fw] = cv2.resize(tiny, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return out


@tool("Artistic", "False color (color map)", [
    P("select", "cmap", "Color map", "Inferno", options=list(COLORMAPS)),
], explain="""
`cv2.applyColorMap` replaces each gray level with a color from a palette - the way thermal cameras and medical
images are displayed. It shows that a "color" is just a lookup from a number.
""")
def colormap(img, cmap):
    return cv2.applyColorMap(gray(img), COLORMAPS[cmap])


@tool("Artistic", "Duotone", [
    P("color", "dark", "Shadow color", "#1E1B4B"),
    P("color", "light", "Highlight color", "#F472B6"),
], explain="""
The gray value of each pixel (0-1) linearly mixes two chosen colors: `dark * (1 - g) + light * g`.
""")
def duotone(img, dark, light):
    g = (gray(img).astype(np.float32) / 255)[..., None]
    d, l = np.array(hex_to_bgr(dark), np.float32), np.array(hex_to_bgr(light), np.float32)
    return (d * (1 - g) + l * g).astype(np.uint8)


@tool("Artistic", "Neon edges", [
    P("range", "thresholds", "Edge thresholds", (60, 150), 0, 400, 5),
    P("slider", "hue", "Neon hue", 150, 0, 179, 1),
    P("slider", "glow", "Glow", 15, 1, 41, 2),
    P("slider", "dim", "Background brightness", 0.15, 0.0, 1.0, 0.05),
], explain="""
Canny edges are colored through HSV, blurred into a glow, and added (`cv2.add`) on top of a darkened photo.
""")
def neon(img, thresholds, hue, glow, dim):
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray(img), (5, 5), 0), *thresholds), None)
    colored = cv2.cvtColor(cv2.merge([np.full_like(edges, hue), np.full_like(edges, 255), edges]), cv2.COLOR_HSV2BGR)
    halo = cv2.GaussianBlur(colored, (odd(glow), odd(glow)), 0)
    return cv2.add(cv2.add((img * dim).astype(np.uint8), halo), colored)


@tool("Artistic", "Stylization (painting)", [
    P("slider", "sigma_s", "Smoothing (sigma_s)", 60, 1, 200, 1),
    P("slider", "sigma_r", "Edge preservation (sigma_r)", 0.45, 0.05, 1.0, 0.05),
], heavy=True, explain="""
`cv2.stylization` is an edge-preserving filter from OpenCV's photo module that produces a watercolor-like look.
""")
def stylization(img, sigma_s, sigma_r):
    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)


@tool("Artistic", "Detail enhance (HDR look)", [
    P("slider", "sigma_s", "Smoothing (sigma_s)", 10, 1, 200, 1),
    P("slider", "sigma_r", "Edge preservation (sigma_r)", 0.15, 0.05, 1.0, 0.05),
], heavy=True, explain="""
`cv2.detailEnhance` boosts fine detail using an edge-preserving filter, giving a crisp "HDR" look.
""")
def detail(img, sigma_s, sigma_r):
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
