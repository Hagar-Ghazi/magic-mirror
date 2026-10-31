import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, gray


def _on_luma(img, fn):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = fn(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


@tool("Histogram", "Histogram equalization", [
    P("select", "mode", "Apply to", "Color (luminance only)", options=["Color (luminance only)", "Grayscale", "Each channel separately"]),
], explain="""
The **histogram** counts how many pixels have each brightness. A dull photo uses only part of the 0-255 range.
`cv2.equalizeHist` spreads the values so the cumulative histogram becomes a straight line, which boosts
contrast. For color photos it is applied to the **luminance (Y of YCrCb)** only; equalizing B, G and R separately
shifts the colors - try it to see why.
""")
def equalize(img, mode):
    if mode == "Grayscale":
        return bgr(cv2.equalizeHist(gray(img)))
    if mode == "Each channel separately":
        return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])
    return _on_luma(img, cv2.equalizeHist)


@tool("Histogram", "CLAHE (adaptive equalization)", [
    P("slider", "clip", "Clip limit", 2.0, 0.5, 10.0, 0.5),
    P("slider", "tiles", "Tile grid size", 8, 2, 24, 1),
    P("select", "mode", "Apply to", "Color (luminance only)", options=["Color (luminance only)", "Grayscale"]),
], explain="""
**Contrast Limited Adaptive Histogram Equalization** equalizes small tiles separately (so dark and bright regions
both improve) and **clips** each tile's histogram to avoid amplifying noise. Created with `cv2.createCLAHE`.
""")
def clahe(img, clip, tiles, mode):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    if mode == "Grayscale":
        return bgr(c.apply(gray(img)))
    return _on_luma(img, c.apply)


@tool("Histogram", "Normalize (contrast stretch)", [
    P("range", "out_range", "Output range", (0, 255), 0, 255, 1),
    P("slider", "clip", "Ignore darkest/brightest (%)", 2.0, 0.0, 5.0, 0.1),
    P("select", "mode", "Apply to", "Color (luminance only)", options=["Color (luminance only)", "Each channel separately"]),
], explain="""
**Min-max normalization** maps the darkest pixel to the new minimum and the brightest to the new maximum
(`cv2.normalize(..., NORM_MINMAX)`), stretching everything in between linearly. Ignoring a small percentage of
extreme pixels (percentile clipping) stops a few outliers from blocking the stretch.
""")
def normalize(img, out_range, clip, mode):
    lo_out, hi_out = out_range

    def stretch(ch):
        lo, hi = np.percentile(ch, [clip, 100 - clip]) if clip > 0 else (ch.min(), ch.max())
        if hi <= lo:
            return ch
        clipped = np.clip(ch, lo, hi).astype(np.float32)
        return cv2.normalize(clipped, None, lo_out, hi_out, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "Each channel separately":
        return cv2.merge([stretch(c) for c in cv2.split(img)])
    return _on_luma(img, stretch)
