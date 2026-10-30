from functools import lru_cache

import cv2
import numpy as np

from core.helpers import overlay_rgba, put_label
from spells.base import Spell

SMOOTH = 0.5  # how much of the previous box to keep (reduces jitter)


@lru_cache(maxsize=None)
def cascade(name):
    return cv2.CascadeClassifier(cv2.data.haarcascades + name)


@lru_cache(maxsize=None)
def sticker(kind):
    """Draw transparent BGRA stickers with OpenCV so no image files are needed."""
    if kind == "Sunglasses":
        img = np.zeros((130, 400, 4), np.uint8)
        for cx in (100, 300):
            cv2.circle(img, (cx, 65), 58, (30, 30, 30, 200), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, 65), 58, (10, 10, 10, 255), 8, cv2.LINE_AA)
            cv2.ellipse(img, (cx - 22, 42), (18, 8), -30, 0, 360, (255, 255, 255, 140), -1, cv2.LINE_AA)
        cv2.line(img, (158, 58), (242, 58), (10, 10, 10, 255), 9, cv2.LINE_AA)
        cv2.line(img, (44, 50), (2, 36), (10, 10, 10, 255), 9, cv2.LINE_AA)
        cv2.line(img, (356, 50), (398, 36), (10, 10, 10, 255), 9, cv2.LINE_AA)
        return img
    if kind == "Mustache":
        img = np.zeros((110, 300, 4), np.uint8)
        cv2.ellipse(img, (95, 50), (78, 30), -12, 0, 360, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.ellipse(img, (205, 50), (78, 30), 12, 0, 360, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (22, 70), 18, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (278, 70), 18, (25, 25, 35, 255), -1, cv2.LINE_AA)
        cv2.ellipse(img, (150, 20), (40, 20), 0, 0, 360, (0, 0, 0, 0), -1)
        return img
    if kind == "Crown":
        img = np.zeros((180, 300, 4), np.uint8)
        pts = np.array([(10, 170), (10, 60), (75, 115), (150, 15), (225, 115), (290, 60), (290, 170)], np.int32)
        cv2.fillPoly(img, [pts], (0, 200, 255, 255), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (0, 130, 200, 255), 5, cv2.LINE_AA)
        for (x, y), color in zip([(10, 60), (150, 15), (290, 60)], [(60, 60, 230), (230, 80, 40), (60, 60, 230)]):
            cv2.circle(img, (x, y), 14, (*color, 255), -1, cv2.LINE_AA)
        cv2.rectangle(img, (10, 140), (290, 170), (0, 160, 230, 255), -1)
        return img
    raise ValueError(kind)


def place(frame, kind, x, y, w, h):
    """Scale a sticker to the face and position it relative to the face box."""
    img = sticker(kind)
    width = {"Sunglasses": 1.0, "Mustache": 0.55, "Crown": 0.95}[kind] * w
    scale = width / img.shape[1]
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    sw, sh = img.shape[1], img.shape[0]
    cx = x + w // 2
    cy = {"Sunglasses": y + int(0.40 * h), "Mustache": y + int(0.70 * h), "Crown": y + int(0.08 * h) - sh // 2}[kind]
    overlay_rgba(frame, img, cx - sw // 2, cy - sh // 2)


class FaceFilter(Spell):
    name = "Face filters"
    icon = "😎"
    tagline = "Sunglasses, mustache, crown - or hide faces for privacy."
    explain = """
**Idea:** find faces with a classic Haar cascade, then stick transparent images on them.

1. **Grayscale + histogram equalization** so faces are found in dim light too (Lab01, Lab03 histograms).
2. **Haar cascade** `detectMultiScale` finds faces; eyes are searched only inside each face (Lab05).
3. **Smoothing** - each new box is blended with the previous one to stop jitter.
4. **Stickers** are drawn with OpenCV shapes and blended using their alpha channel (Lab03 arithmetic + masking).
5. **Privacy modes** - *Pixelate* shrinks and re-enlarges the face (Lab02 resize), *Blur* uses a big Gaussian kernel (Lab04).
"""

    def sidebar(self, ui, mode):
        p = {}
        p["filters"] = ui.multiselect(
            "Filters", ["Sunglasses", "Mustache", "Crown", "Pixelate", "Blur"], ["Sunglasses", "Crown"], key="face_filters"
        )
        p["boxes"] = ui.toggle("Show detection boxes", False, key="face_boxes")
        p["scale"] = ui.slider("Scale factor", 1.05, 1.5, 1.15, step=0.05, key="face_scale")
        p["neighbors"] = ui.slider("Min neighbors", 2, 12, 6, key="face_nb")
        p["blocks"] = ui.slider("Pixelate blocks", 4, 30, 12, key="face_blocks")
        return p

    def process(self, frame, p, state):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        faces = cascade("haarcascade_frontalface_default.xml").detectMultiScale(
            eq, scaleFactor=p["scale"], minNeighbors=p["neighbors"], minSize=(60, 60)
        )
        faces = sorted([tuple(int(v) for v in f) for f in faces], key=lambda f: f[0])

        prev = state.get("faces", [])
        if len(prev) == len(faces):
            faces = [
                tuple(int(SMOOTH * a + (1 - SMOOTH) * b) for a, b in zip(old, new)) for old, new in zip(prev, faces)
            ]
        state["faces"] = faces

        detections = frame.copy()
        result = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eyes = cascade("haarcascade_eye.xml").detectMultiScale(eq[y:y + h // 2, x:x + w], 1.1, 8)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(detections, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 200, 0), 2)

            roi = result[y:y + h, x:x + w]
            if "Pixelate" in p["filters"]:
                small = cv2.resize(roi, (p["blocks"], p["blocks"]), interpolation=cv2.INTER_LINEAR)
                result[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            if "Blur" in p["filters"]:
                k = (w // 3) | 1
                result[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
            for kind in ("Crown", "Sunglasses", "Mustache"):
                if kind in p["filters"]:
                    place(result, kind, x, y, w, h)
            if p["boxes"]:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not faces:
            put_label(result, "No face found - face the camera", scale=0.6)
        else:
            put_label(result, f"Faces: {len(faces)}", org=(10, result.shape[0] - 15), scale=0.6)

        steps = [("Original", frame), ("Grayscale", gray), ("Equalized histogram", eq), ("Faces and eyes", detections)]
        return result, steps
