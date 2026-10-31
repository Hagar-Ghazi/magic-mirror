import os
from functools import lru_cache

import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
EMOJI_DIR = os.path.join(ASSETS, "emoji")


def bgr(img):
    """Make sure an image is 3-channel BGR uint8."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def pt(img, x_pct, y_pct):
    """Percent coordinates (0-100) to pixel coordinates."""
    h, w = img.shape[:2]
    return int(round(x_pct / 100 * (w - 1))), int(round(y_pct / 100 * (h - 1)))


def odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def overlay_rgba(frame, rgba, x, y):
    """Alpha-blend a BGRA image onto frame with its top-left at (x, y), clipped at the edges."""
    fh, fw = frame.shape[:2]
    sh, sw = rgba.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, fw), min(y + sh, fh)
    if x0 >= x1 or y0 >= y1:
        return frame
    sticker = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = sticker[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * sticker[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame


def fit_to(img, shape):
    """Resize img to the height/width of `shape` (for blending two images)."""
    return cv2.resize(bgr(img), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=None)
def cascade(name):
    return cv2.CascadeClassifier(cv2.data.haarcascades + name)


def detect_faces(img, scale=1.1, neighbors=5, min_size=40):
    g = cv2.equalizeHist(gray(img))
    faces = cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        g, scaleFactor=scale, minNeighbors=neighbors, minSize=(min_size, min_size)
    )
    return [tuple(int(v) for v in f) for f in faces]


def emoji_names():
    if not os.path.isdir(EMOJI_DIR):
        return []
    return sorted(os.path.splitext(f)[0].replace("_", " ").capitalize() for f in os.listdir(EMOJI_DIR))


@lru_cache(maxsize=None)
def load_emoji(name):
    path = os.path.join(EMOJI_DIR, name.lower().replace(" ", "_") + ".png")
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def shape_mask(shape_hw, kind, cx, cy, w_pct, h_pct, feather=0):
    """White-on-black mask of a centered shape. Sizes are percentages of the image size."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    center = (int(cx / 100 * w), int(cy / 100 * h))
    half = (max(1, int(w_pct / 200 * w)), max(1, int(h_pct / 200 * h)))
    if kind == "Rectangle":
        cv2.rectangle(mask, (center[0] - half[0], center[1] - half[1]), (center[0] + half[0], center[1] + half[1]), 255, -1)
    elif kind == "Circle":
        cv2.circle(mask, center, min(half), 255, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(mask, center, half, 0, 0, 360, 255, -1, cv2.LINE_AA)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def put_label(img, text, org=(10, 30), color=(255, 255, 255), bg=(0, 0, 0), scale=0.7):
    """Text on a filled box, readable on any image (used for status messages)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, scale * img.shape[1] / 800)
    th = max(1, int(scale * 2))
    (w, h), base = cv2.getTextSize(text, font, scale, th)
    x, y = org[0], max(org[1], h + 10)
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
    return img


def blend_mask(a, b, mask):
    """Where mask is white take a, where black take b (soft masks blend smoothly)."""
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return (a.astype(np.float32) * m + b.astype(np.float32) * (1 - m)).astype(np.uint8)
