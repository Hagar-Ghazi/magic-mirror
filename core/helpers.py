import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# HSV ranges (OpenCV hue is 0-179). Red wraps around 0, so it needs two ranges.
COLOR_PRESETS = {
    "Red": [(0, 10), (170, 179)],
    "Blue": [(94, 126)],
    "Green": [(36, 86)],
    "Yellow": [(20, 35)],
}


def build_ranges(preset, hue_range, s_min, v_min):
    """Turn a preset (or custom hue range) plus S/V minimums into HSV bounds."""
    hues = [hue_range] if preset == "Custom" else COLOR_PRESETS[preset]
    return [((h0, s_min, v_min), (h1, 255, 255)) for h0, h1 in hues]


def hsv_mask(frame, ranges):
    """Binary mask of pixels inside any of the HSV ranges."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    return mask


def clean_mask(mask, kernel_size=5):
    """Remove speckles (opening) and fill small holes (dilation)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.dilate(mask, kernel, iterations=1)


def largest_contour(mask, min_area=500):
    """Biggest external contour, or None if nothing is large enough."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return biggest if cv2.contourArea(biggest) >= min_area else None


def contour_center(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def overlay_rgba(frame, rgba, x, y):
    """Alpha-blend an RGBA (BGRA) sticker onto frame at (x, y), clipping at the edges."""
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


def put_label(img, text, org=(10, 30), color=(255, 255, 255), bg=(0, 0, 0), scale=0.7):
    """Text with a filled background box so it stays readable on any frame."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)
    return img


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r
