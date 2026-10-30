from collections import deque

import cv2
import numpy as np

from core.helpers import clean_mask, contour_center, hex_to_bgr, hsv_mask, largest_contour, put_label
from spells.base import Spell, action_button, color_controls, take_action

SMOOTHING = 5


class AirDraw(Spell):
    name = "Air drawing"
    icon = "✍️"
    tagline = "Draw in the air with a brightly colored pen cap or ball."
    explain = """
**Idea:** find the pen tip in every frame, connect its positions, and paint those lines on a
transparent canvas laid over the video.

1. **HSV mask** on the pen color, then **clean** it (Lab05 color tracking, Lab04 morphology).
2. **Largest contour** = the pen; its **centroid** from image moments is the tip (Lab05 contours).
3. **Smoothing** - average the last 5 positions so the line does not shake.
4. **Strokes** - if the pen disappears the current stroke ends, so you can lift the pen.
5. **Overlay** - the canvas is copied onto the frame wherever it has paint (Lab03 masking).
"""

    def sidebar(self, ui, mode):
        p = {"ranges": color_controls(ui, "draw", "Blue")}
        p["kernel"] = ui.slider("Mask cleanup kernel", 1, 15, 5, step=2, key="draw_k")
        p["min_area"] = ui.slider("Min pen size (px²)", 50, 3000, 300, step=50, key="draw_area")
        p["brush"] = hex_to_bgr(ui.color_picker("Brush color", "#FF2E88", key="draw_color"))
        p["thickness"] = ui.slider("Brush thickness", 2, 30, 8, key="draw_thick")
        p["drawing"] = ui.toggle("Pen down (drawing on)", True, key="draw_on")
        c1, c2 = ui.columns(2)
        p["actions"] = {
            "undo": action_button(c1, "Undo", "draw_undo"),
            "clear": action_button(c2, "Clear", "draw_clear"),
        }
        return p

    def process(self, frame, p, state):
        strokes = state.setdefault("strokes", [])
        current = state.setdefault("current", None)
        recent = state.setdefault("recent", deque(maxlen=SMOOTHING))

        if take_action(state, p, "clear"):
            strokes.clear()
            state["current"] = current = None
        if take_action(state, p, "undo") and strokes:
            strokes.pop()

        mask = hsv_mask(frame, p["ranges"])
        clean = clean_mask(mask, p["kernel"])
        pen = largest_contour(clean, p["min_area"])
        tracked = frame.copy()

        center = contour_center(pen) if pen is not None else None
        if center is None:
            # pen lifted or lost: finish the stroke
            if current and len(current["points"]) > 1:
                strokes.append(current)
            state["current"] = current = None
            recent.clear()
        else:
            recent.append(center)
            smooth = tuple(int(v) for v in np.mean(recent, axis=0))
            cv2.drawContours(tracked, [pen], -1, (0, 255, 0), 2)
            cv2.circle(tracked, smooth, 8, (0, 255, 255), -1)
            if p["drawing"]:
                if current is None:
                    state["current"] = current = {"points": [], "color": p["brush"], "thickness": p["thickness"]}
                current["points"].append(smooth)

        canvas = np.zeros_like(frame)
        for stroke in strokes + ([current] if current else []):
            pts = np.array(stroke["points"], np.int32)
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, stroke["color"], stroke["thickness"], cv2.LINE_AA)

        paint = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
        result = tracked.copy()
        result[paint] = canvas[paint]
        if center is None:
            put_label(result, "Show the pen to draw", scale=0.6)
        put_label(result, f"Strokes: {len(strokes)}", org=(10, result.shape[0] - 15), scale=0.6)

        steps = [
            ("Original", frame),
            ("HSV mask", mask),
            ("Cleaned mask", clean),
            ("Pen tracking", tracked),
            ("Canvas", canvas),
        ]
        return result, steps
