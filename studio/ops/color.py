import itertools

import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, hex_to_bgr

SPACES = {
    "HSV": (cv2.COLOR_BGR2HSV, ["Hue", "Saturation", "Value"]),
    "HLS": (cv2.COLOR_BGR2HLS, ["Hue", "Lightness", "Saturation"]),
    "LAB": (cv2.COLOR_BGR2LAB, ["Lightness", "A (green-red)", "B (blue-yellow)"]),
    "YCrCb": (cv2.COLOR_BGR2YCrCb, ["Luma Y", "Cr (red diff)", "Cb (blue diff)"]),
    "BGR": (None, ["Blue", "Green", "Red"]),
}


def _label(img, text):
    out = img.copy()
    s = max(0.5, img.shape[1] / 500)
    cv2.putText(out, text, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), int(4 * s), cv2.LINE_AA)
    cv2.putText(out, text, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return out


def _grid(tiles):
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


@tool("Color", "Grayscale", [
    P("select", "method", "Method", "Weighted (cvtColor)", options=["Weighted (cvtColor)", "Average", "Lightness", "Single channel"]),
    P("select", "channel", "Channel", "Green", options=["Blue", "Green", "Red"], when=lambda v: v["method"] == "Single channel"),
], explain="""
`cv2.cvtColor(..., COLOR_BGR2GRAY)` uses `0.299 R + 0.587 G + 0.114 B` because the eye is most sensitive to green.
Compare it with a plain **average** of the channels, the **lightness** `(max + min) / 2`, or a single channel.
""")
def grayscale(img, method, channel="Green"):
    if method == "Weighted (cvtColor)":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif method == "Average":
        g = img.mean(axis=2).astype(np.uint8)
    elif method == "Lightness":
        g = ((img.max(axis=2).astype(np.uint16) + img.min(axis=2)) // 2).astype(np.uint8)
    else:
        g = img[..., ["Blue", "Green", "Red"].index(channel)]
    return bgr(g)


@tool("Color", "Color space explorer", [
    P("select", "space", "Color space", "HSV", options=list(SPACES)),
    P("select", "view", "Show", "All channels (grid)", options=["All channels (grid)", "Channel 1", "Channel 2", "Channel 3", "Raw values shown as BGR"]),
], explain="""
The same pixels can be described in different **color spaces**. `cv2.cvtColor` converts BGR to HSV, HLS, LAB or
YCrCb; each channel is shown as a grayscale image. *Raw values shown as BGR* displays the converted numbers as if
they were still BGR, which is exactly the bug you get when you forget to convert back.
""")
def color_space(img, space, view):
    code, names = SPACES[space]
    conv = cv2.cvtColor(img, code) if code is not None else img.copy()
    chans = cv2.split(conv)
    if view == "Raw values shown as BGR":
        return conv
    if view.startswith("Channel"):
        i = int(view[-1]) - 1
        return _label(bgr(chans[i]), f"{space}: {names[i]}")
    tiles = [_label(img, "Original")] + [_label(bgr(c), f"{space}: {n}") for c, n in zip(chans, names)]
    return _grid(tiles)


@tool("Color", "Channels: keep and swap", [
    P("toggle", "keep_b", "Keep blue", True),
    P("toggle", "keep_g", "Keep green", True),
    P("toggle", "keep_r", "Keep red", True),
    P("select", "order", "Channel order", "R G B", options=["B G R (normal)"] + [" ".join(p) for p in itertools.permutations("BGR") if p != ("B", "G", "R")]),
], explain="""
`cv2.split` gives the B, G and R planes; turning one off sets it to zero, and `cv2.merge` puts them back in any
order. Swapping to **R G B** shows why OpenCV images look blue-tinted in other libraries: they expect RGB.
""")
def channels(img, keep_b, keep_g, keep_r, order):
    b, g, r = cv2.split(img)
    zero = np.zeros_like(b)
    planes = {"B": b if keep_b else zero, "G": g if keep_g else zero, "R": r if keep_r else zero}
    names = order.split(" ")[:3]
    return cv2.merge([planes[n] for n in names])


@tool("Color", "Split channels (grid)", [
    P("select", "style", "Show channels as", "Tinted color", options=["Tinted color", "Grayscale"]),
], explain="""
`cv2.split` separates the image into its blue, green and red planes. In grayscale, bright means "a lot of this
color". Tinted view puts each plane back into its own channel with the other two set to zero.
""")
def split_channels(img, style):
    b, g, r = cv2.split(img)
    z = np.zeros_like(b)
    if style == "Tinted color":
        tiles = [cv2.merge([b, z, z]), cv2.merge([z, g, z]), cv2.merge([z, z, r])]
    else:
        tiles = [bgr(b), bgr(g), bgr(r)]
    tiles = [_label(img, "Original")] + [_label(t, n) for t, n in zip(tiles, ["Blue", "Green", "Red"])]
    return _grid(tiles)


@tool("Color", "Color range (HSV mask)", [
    P("range", "hue", "Hue range", (35, 85), 0, 179, 1, help="Red 0-10 or 170-179, yellow 20-35, green 35-85, blue 90-130"),
    P("range", "sat", "Saturation range", (60, 255), 0, 255, 1),
    P("range", "val", "Value range", (40, 255), 0, 255, 1),
    P("select", "mode", "Result", "Color splash (rest gray)", options=[
        "Color splash (rest gray)", "Keep only this color", "Remove this color", "Show mask", "Recolor"]),
    P("slider", "new_hue", "New hue", 150, 0, 179, 1, when=lambda v: v["mode"] == "Recolor"),
], explain="""
`cv2.inRange` on the HSV image marks every pixel whose hue, saturation and value fall inside the ranges.
That binary mask then drives `bitwise_and` (keep), its inverse (remove), a **color splash** (mask selects
between the color image and its grayscale copy) or a hue change only inside the mask.
""")
def color_range(img, hue, sat, val, mode, new_hue=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))
    if mode == "Show mask":
        return bgr(mask)
    if mode == "Keep only this color":
        return cv2.bitwise_and(img, img, mask=mask)
    if mode == "Remove this color":
        return cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    if mode == "Recolor":
        out = hsv.copy()
        out[..., 0][mask > 0] = new_hue
        return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    gray = bgr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return np.where(mask[..., None] > 0, img, gray)


@tool("Color", "Fill with solid color", [
    P("color", "color", "Color", "#7C3AED"),
    P("slider", "opacity", "Opacity", 0.3, 0.0, 1.0, 0.05),
], explain="""
Creates a solid image with `np.full` and blends it over the photo with `cv2.addWeighted` - a color wash or tint.
""")
def color_fill(img, color, opacity):
    solid = np.full_like(img, hex_to_bgr(color))
    return cv2.addWeighted(solid, opacity, img, 1 - opacity, 0)
