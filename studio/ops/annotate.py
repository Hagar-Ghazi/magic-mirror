from functools import lru_cache

import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import detect_faces, hex_to_bgr, load_emoji, overlay_rgba, pt, put_label

FONTS = {
    "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "Duplex": cv2.FONT_HERSHEY_DUPLEX,
    "Complex": cv2.FONT_HERSHEY_COMPLEX,
    "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "Plain": cv2.FONT_HERSHEY_PLAIN,
    "Script simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "Script complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}
ANCHORS = ["Top-left corner", "Center"]
BORDER_TYPES = {
    "Constant color": cv2.BORDER_CONSTANT,
    "Reflect": cv2.BORDER_REFLECT,
    "Reflect 101": cv2.BORDER_REFLECT_101,
    "Replicate": cv2.BORDER_REPLICATE,
    "Wrap": cv2.BORDER_WRAP,
}


def _rotate_rgba(rgba, angle):
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


@tool("Annotate", "Text", [
    P("text", "text", "Text", "Hello, OpenCV!", help="Hershey fonts support English letters, digits and symbols"),
    P("slider", "x", "X position (%)", 50, 0, 100, 1),
    P("slider", "y", "Y position (%)", 90, 0, 100, 1),
    P("select", "anchor", "Position refers to", "Center", options=ANCHORS),
    P("select", "font", "Font", "Duplex", options=list(FONTS)),
    P("toggle", "italic", "Italic", False),
    P("slider", "size", "Size (% of image width)", 6.0, 1.0, 30.0, 0.5),
    P("slider", "thickness", "Thickness", 2, 1, 20, 1),
    P("color", "color", "Text color", "#FFFFFF"),
    P("toggle", "outline", "Outline", True),
    P("color", "outline_color", "Outline color", "#000000", when=lambda v: v["outline"]),
    P("toggle", "box", "Background box", False),
    P("color", "box_color", "Box color", "#7C3AED", when=lambda v: v["box"]),
    P("slider", "box_opacity", "Box opacity", 0.7, 0.0, 1.0, 0.05, when=lambda v: v["box"]),
], explain="""
`cv2.putText(img, text, origin, font, scale, color, thickness)` draws text whose **origin is the bottom-left
corner of the baseline**. `cv2.getTextSize` measures the text first, so it can be centered, given a background
box (blended with `addWeighted` for transparency) and an outline (the same text drawn thicker underneath).
""")
def text(img, text, x, y, anchor, font, italic, size, thickness, color, outline, box, outline_color="#000000",
         box_color="#7C3AED", box_opacity=0.7):
    if not text:
        return img
    face = FONTS[font] | (cv2.FONT_ITALIC if italic else 0)
    (tw1, th1), _ = cv2.getTextSize("M", face, 1.0, thickness)
    scale = size / 100 * img.shape[1] / max(th1, 1) * 0.7
    (tw, th), base = cv2.getTextSize(text, face, scale, thickness)
    px, py = pt(img, x, y)
    if anchor == "Center":
        px, py = px - tw // 2, py + th // 2
    out = img
    if box:
        pad = max(4, th // 3)
        layer = out.copy()
        cv2.rectangle(layer, (px - pad, py - th - pad), (px + tw + pad, py + base + pad), hex_to_bgr(box_color), -1)
        out = cv2.addWeighted(layer, box_opacity, out, 1 - box_opacity, 0)
    if outline:
        cv2.putText(out, text, (px, py), face, scale, hex_to_bgr(outline_color), thickness + max(2, thickness), cv2.LINE_AA)
    cv2.putText(out, text, (px, py), face, scale, hex_to_bgr(color), thickness, cv2.LINE_AA)
    return out


@tool("Annotate", "Emoji", [
    P("emoji", "emoji", "Emoji", "Smiling face with sunglasses"),
    P("slider", "x", "Center X (%)", 50, 0, 100, 1),
    P("slider", "y", "Center Y (%)", 30, 0, 100, 1),
    P("slider", "size", "Size (% of image width)", 20, 2, 100, 1),
    P("slider", "angle", "Rotation (degrees)", 0, -180, 180, 5),
    P("slider", "opacity", "Opacity", 1.0, 0.1, 1.0, 0.05),
    P("toggle", "mirror", "Mirror", False),
], explain="""
Emoji are PNG images with a 4th **alpha channel** (transparency), loaded with `cv2.IMREAD_UNCHANGED`. They are
resized, optionally flipped and rotated (`warpAffine` with a transparent border), then **alpha-blended**:
`out = alpha * emoji + (1 - alpha) * photo`.
""")
def emoji(img, emoji, x, y, size, angle, opacity, mirror):
    rgba = load_emoji(emoji)
    if rgba is None:
        return put_label(img.copy(), "Emoji image not found")
    side = max(4, int(size / 100 * img.shape[1]))
    rgba = cv2.resize(rgba, (side, side), interpolation=cv2.INTER_AREA if side < rgba.shape[0] else cv2.INTER_CUBIC)
    if mirror:
        rgba = cv2.flip(rgba, 1)
    if angle:
        rgba = _rotate_rgba(rgba, angle)
    if opacity < 1:
        rgba = rgba.copy()
        rgba[..., 3] = (rgba[..., 3] * opacity).astype(np.uint8)
    cx, cy = pt(img, x, y)
    return overlay_rgba(img, rgba, cx - rgba.shape[1] // 2, cy - rgba.shape[0] // 2)


@lru_cache(maxsize=None)
def _sticker(kind):
    if kind == "Sunglasses":
        s = np.zeros((130, 400, 4), np.uint8)
        for cx in (100, 300):
            cv2.circle(s, (cx, 65), 58, (30, 30, 30, 210), -1, cv2.LINE_AA)
            cv2.circle(s, (cx, 65), 58, (10, 10, 10, 255), 8, cv2.LINE_AA)
            cv2.ellipse(s, (cx - 22, 42), (18, 8), -30, 0, 360, (255, 255, 255, 140), -1, cv2.LINE_AA)
        cv2.line(s, (158, 58), (242, 58), (10, 10, 10, 255), 9, cv2.LINE_AA)
        return s
    if kind == "Mustache":
        s = np.zeros((110, 300, 4), np.uint8)
        cv2.ellipse(s, (95, 50), (78, 30), -12, 0, 360, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.ellipse(s, (205, 50), (78, 30), 12, 0, 360, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.circle(s, (22, 70), 18, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.circle(s, (278, 70), 18, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.ellipse(s, (150, 20), (40, 20), 0, 0, 360, (0, 0, 0, 0), -1)
        return s
    s = np.zeros((180, 300, 4), np.uint8)
    pts = np.array([(10, 170), (10, 60), (75, 115), (150, 15), (225, 115), (290, 60), (290, 170)], np.int32)
    cv2.fillPoly(s, [pts], (0, 200, 255, 255), cv2.LINE_AA)
    cv2.polylines(s, [pts], True, (0, 130, 200, 255), 5, cv2.LINE_AA)
    for (x, y), c in zip([(10, 60), (150, 15), (290, 60)], [(60, 60, 230), (230, 80, 40), (60, 60, 230)]):
        cv2.circle(s, (x, y), 14, (*c, 255), -1, cv2.LINE_AA)
    return s


@tool("Annotate", "Face stickers", [
    P("select", "kind", "Sticker", "Sunglasses", options=["Sunglasses", "Mustache", "Crown"]),
    P("slider", "size", "Size", 1.0, 0.5, 1.8, 0.05),
    P("slider", "neighbors", "Detection strictness", 5, 2, 12, 1),
], explain="""
Faces are found with a **Haar cascade**; the sticker (drawn with OpenCV shapes into a transparent image) is scaled
to each face width and placed relative to the face box: eyes at about 40% of its height, mouth at 70%, crown above.
""")
def face_stickers(img, kind, size, neighbors):
    faces = detect_faces(img, neighbors=neighbors)
    if not faces:
        return put_label(img.copy(), "No face found - try lower strictness")
    out = img.copy()
    for x, y, w, h in faces:
        s = _sticker(kind)
        width = {"Sunglasses": 1.0, "Mustache": 0.55, "Crown": 0.95}[kind] * w * size
        k = width / s.shape[1]
        s = cv2.resize(s, (max(1, int(s.shape[1] * k)), max(1, int(s.shape[0] * k))), interpolation=cv2.INTER_AREA)
        cy = {"Sunglasses": y + 0.40 * h, "Mustache": y + 0.72 * h, "Crown": y + 0.05 * h - s.shape[0] / 2}[kind]
        overlay_rgba(out, s, int(x + w / 2 - s.shape[1] / 2), int(cy - s.shape[0] / 2))
    return out


@tool("Annotate", "Shapes", [
    P("select", "shape", "Shape", "Rectangle", options=["Rectangle", "Circle", "Ellipse", "Line", "Arrow", "Star"]),
    P("slider", "x1", "Start / center X (%)", 30, 0, 100, 1),
    P("slider", "y1", "Start / center Y (%)", 30, 0, 100, 1),
    P("slider", "x2", "End X / size (%)", 70, 0, 100, 1),
    P("slider", "y2", "End Y / size (%)", 70, 0, 100, 1),
    P("color", "color", "Color", "#22D3EE"),
    P("slider", "thickness", "Thickness", 4, 1, 40, 1),
    P("toggle", "filled", "Filled", False),
    P("slider", "opacity", "Opacity", 1.0, 0.1, 1.0, 0.05),
], explain="""
OpenCV draws directly into the pixel array: `cv2.rectangle`, `cv2.circle`, `cv2.ellipse`, `cv2.line`,
`cv2.arrowedLine`, and `cv2.fillPoly` / `cv2.polylines` for any polygon (the star). Thickness `-1` fills the shape.
Drawing on a copy and blending with `addWeighted` gives transparency.
""")
def shapes(img, shape, x1, y1, x2, y2, color, thickness, filled, opacity):
    layer = img.copy()
    c = hex_to_bgr(color)
    t = -1 if filled and shape not in ("Line", "Arrow") else thickness
    p1, p2 = pt(img, x1, y1), pt(img, x2, y2)
    if shape == "Rectangle":
        cv2.rectangle(layer, p1, p2, c, t, cv2.LINE_AA)
    elif shape == "Circle":
        cv2.circle(layer, p1, max(1, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))), c, t, cv2.LINE_AA)
    elif shape == "Ellipse":
        cv2.ellipse(layer, p1, (abs(p2[0] - p1[0]) or 1, abs(p2[1] - p1[1]) or 1), 0, 0, 360, c, t, cv2.LINE_AA)
    elif shape == "Line":
        cv2.line(layer, p1, p2, c, thickness, cv2.LINE_AA)
    elif shape == "Arrow":
        cv2.arrowedLine(layer, p1, p2, c, thickness, cv2.LINE_AA, tipLength=0.15)
    else:
        r = max(4, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        ang = np.deg2rad(np.arange(10) * 36 - 90)
        rad = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
        pts = np.stack([p1[0] + rad * np.cos(ang), p1[1] + rad * np.sin(ang)], 1).astype(np.int32)
        if filled:
            cv2.fillPoly(layer, [pts], c, cv2.LINE_AA)
        else:
            cv2.polylines(layer, [pts], True, c, thickness, cv2.LINE_AA)
    return cv2.addWeighted(layer, opacity, img, 1 - opacity, 0) if opacity < 1 else layer


@tool("Annotate", "Border and padding", [
    P("select", "kind", "Border type", "Constant color", options=list(BORDER_TYPES)),
    P("color", "color", "Color", "#FFFFFF", when=lambda v: v["kind"] == "Constant color"),
    P("slider", "top", "Top (% of height)", 4, 0, 50, 1),
    P("slider", "bottom", "Bottom (% of height)", 4, 0, 50, 1),
    P("slider", "left", "Left (% of width)", 4, 0, 50, 1),
    P("slider", "right", "Right (% of width)", 4, 0, 50, 1),
], explain="""
`cv2.copyMakeBorder` adds pixels around the image. **Constant** fills with a color (a photo frame or a
polaroid with a thicker bottom), **Reflect** mirrors the edge, **Replicate** repeats the last row/column and
**Wrap** continues from the opposite side. The same modes decide what blur and warp functions do at the edges.
""")
def border(img, kind, top, bottom, left, right, color="#FFFFFF"):
    h, w = img.shape[:2]
    sizes = [int(top / 100 * h), int(bottom / 100 * h), int(left / 100 * w), int(right / 100 * w)]
    if kind != "Constant color":  # reflect/wrap cannot be larger than the image itself
        sizes = [min(s, (h if i < 2 else w) - 1) for i, s in enumerate(sizes)]
    return cv2.copyMakeBorder(img, *sizes, BORDER_TYPES[kind], value=hex_to_bgr(color))


@tool("Annotate", "Watermark", [
    P("text", "text", "Text", "(c) My Photo Studio"),
    P("select", "layout", "Layout", "Bottom-right corner", options=["Bottom-right corner", "Center", "Tiled diagonal"]),
    P("slider", "opacity", "Opacity", 0.35, 0.05, 1.0, 0.05),
    P("slider", "size", "Size", 3.0, 1.0, 15.0, 0.5),
    P("color", "color", "Color", "#FFFFFF"),
], explain="""
Text is drawn on a separate layer and blended with `cv2.addWeighted` so it stays see-through. The tiled version
draws a grid of text on a larger canvas, rotates it 30 degrees with `warpAffine` and crops it back.
""")
def watermark(img, text, layout, opacity, size, color):
    h, w = img.shape[:2]
    face = cv2.FONT_HERSHEY_DUPLEX
    scale = size / 100 * w / 22
    th = max(1, int(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, face, scale, th)
    if layout == "Tiled diagonal":
        side = int(np.hypot(w, h))
        layer = np.zeros((side, side, 3), np.uint8)
        mask = np.zeros((side, side), np.uint8)
        for yy in range(tht * 2, side, tht * 4):
            for xx in range((yy // (tht * 4)) % 2 * tw // 2, side, tw + tht * 3):
                cv2.putText(layer, text, (xx, yy), face, scale, hex_to_bgr(color), th, cv2.LINE_AA)
                cv2.putText(mask, text, (xx, yy), face, scale, 255, th, cv2.LINE_AA)
        m = cv2.getRotationMatrix2D((side / 2, side / 2), 30, 1)
        layer, mask = cv2.warpAffine(layer, m, (side, side)), cv2.warpAffine(mask, m, (side, side))
        oy, ox = (side - h) // 2, (side - w) // 2
        layer, mask = layer[oy:oy + h, ox:ox + w], mask[oy:oy + h, ox:ox + w]
    else:
        layer = np.zeros_like(img)
        mask = np.zeros((h, w), np.uint8)
        org = (w - tw - int(0.03 * w), h - int(0.04 * h)) if layout.startswith("Bottom") else ((w - tw) // 2, (h + tht) // 2)
        cv2.putText(layer, text, org, face, scale, hex_to_bgr(color), th, cv2.LINE_AA)
        cv2.putText(mask, text, org, face, scale, 255, th, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255 * opacity)[..., None]
    return (layer * a + img * (1 - a)).astype(np.uint8)


@tool("Annotate", "Frame shape (rounded, circle)", [
    P("select", "shape", "Shape", "Rounded rectangle", options=["Rounded rectangle", "Circle", "Ellipse"]),
    P("slider", "radius", "Corner radius (%)", 8, 1, 50, 1, when=lambda v: v["shape"] == "Rounded rectangle"),
    P("color", "background", "Background", "#FFFFFF"),
], explain="""
A mask of the frame shape is drawn (rectangles plus four circles make rounded corners). Pixels outside the mask are
replaced by the background color - the same **masking** idea as `bitwise_and` with a mask, with anti-aliased edges.
""")
def frame_shape(img, shape, background, radius=8):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if shape == "Rounded rectangle":
        r = int(radius / 100 * min(h, w))
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        for cx, cy in [(r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)]:
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    elif shape == "Circle":
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255)[..., None]
    bg = np.full_like(img, hex_to_bgr(background), dtype=np.float32)
    return (img * a + bg * (1 - a)).astype(np.uint8)
