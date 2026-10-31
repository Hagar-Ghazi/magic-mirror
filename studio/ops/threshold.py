import cv2

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, gray, odd, put_label

TYPES = {
    "Binary": cv2.THRESH_BINARY,
    "Binary inverted": cv2.THRESH_BINARY_INV,
    "Truncate": cv2.THRESH_TRUNC,
    "To zero": cv2.THRESH_TOZERO,
    "To zero inverted": cv2.THRESH_TOZERO_INV,
}
MORPH = {
    "Erode": cv2.MORPH_ERODE,
    "Dilate": cv2.MORPH_DILATE,
    "Open (erode then dilate)": cv2.MORPH_OPEN,
    "Close (dilate then erode)": cv2.MORPH_CLOSE,
    "Gradient (outline)": cv2.MORPH_GRADIENT,
    "Top hat": cv2.MORPH_TOPHAT,
    "Black hat": cv2.MORPH_BLACKHAT,
}
SHAPES = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}


@tool("Threshold and morphology", "Threshold (binary)", [
    P("select", "kind", "Type", "Binary", options=list(TYPES)),
    P("select", "method", "Threshold value", "Manual", options=["Manual", "Otsu (automatic)", "Triangle (automatic)"]),
    P("slider", "value", "Threshold", 127, 0, 255, 1, when=lambda v: v["method"] == "Manual"),
    P("slider", "blur", "Blur first (kernel)", 1, 1, 15, 2),
], explain="""
`cv2.threshold` turns a gray image into two levels: pixels above the threshold become 255, the rest 0
(**binary**). The other types keep or cut part of the range. **Otsu** picks the threshold automatically by
finding the value that best separates the two peaks of the histogram; **Triangle** works well when one peak dominates.
""")
def threshold(img, kind, method, blur, value=127):
    g = gray(img)
    if blur > 1:
        g = cv2.GaussianBlur(g, (odd(blur), odd(blur)), 0)
    flag = TYPES[kind]
    if method.startswith("Otsu"):
        flag |= cv2.THRESH_OTSU
    elif method.startswith("Triangle"):
        flag |= cv2.THRESH_TRIANGLE
    t, out = cv2.threshold(g, value, 255, flag)
    out = bgr(out)
    if method != "Manual":
        put_label(out, f"Chosen threshold: {t:.0f}", bg=(90, 40, 120))
    return out


@tool("Threshold and morphology", "Adaptive threshold", [
    P("select", "method", "Method", "Gaussian", options=["Mean", "Gaussian"]),
    P("slider", "block", "Block size", 21, 3, 151, 2),
    P("slider", "c", "Constant C", 5, -30, 30, 1),
    P("toggle", "invert", "Invert", False),
], explain="""
One global threshold fails when lighting is uneven (a shadow on a page). `cv2.adaptiveThreshold` computes a
separate threshold for **every pixel** from its neighbourhood (mean or Gaussian-weighted mean of a block),
minus a constant C. Great for documents and handwriting.
""")
def adaptive(img, method, block, c, invert):
    m = cv2.ADAPTIVE_THRESH_MEAN_C if method == "Mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    t = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return bgr(cv2.adaptiveThreshold(gray(img), 255, m, t, odd(block), c))


@tool("Threshold and morphology", "Morphology", [
    P("select", "op", "Operation", "Open (erode then dilate)", options=list(MORPH)),
    P("select", "shape", "Kernel shape", "Ellipse", options=list(SHAPES)),
    P("slider", "size", "Kernel size", 5, 1, 41, 2),
    P("slider", "iterations", "Iterations", 1, 1, 10, 1),
    P("select", "source", "Apply to", "Binary (Otsu) version", options=["Binary (Otsu) version", "Image as is"]),
], explain="""
Morphology changes **shapes** in (usually binary) images using a small structuring element:

- **Erode** shrinks white regions and removes small specks; **dilate** grows them and fills gaps.
- **Open** = erode then dilate: removes noise but keeps object size. **Close** = dilate then erode: fills holes.
- **Gradient** = dilate minus erode: the outline. **Top hat** / **black hat** extract small bright / dark details.
""")
def morphology(img, op, shape, size, iterations, source):
    src = img
    if source.startswith("Binary"):
        _, src = cv2.threshold(gray(img), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(SHAPES[shape], (size, size))
    return bgr(cv2.morphologyEx(src, MORPH[op], kernel, iterations=iterations))
