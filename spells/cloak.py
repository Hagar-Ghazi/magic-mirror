import math
import time

import cv2
import numpy as np

from core.helpers import clean_mask, hsv_mask, put_label
from core.preprocess import decode_image
from spells.base import Spell, action_button, color_controls, take_action

BG_FRAMES = 30
COUNTDOWN = 3


class Cloak(Spell):
    name = "Invisibility cloak"
    icon = "🧥"
    tagline = "Hold a solid-colored cloth and everything behind it shows through."
    explain = """
**Idea:** remember what the scene looks like *without* you, then wherever the cloak color
appears, paste pixels from that remembered background.

1. **Capture background** - median of 30 frames removes noise and anything that moved (Lab05: video).
2. **HSV mask** - `cv2.inRange` on the cloak color. Red wraps around hue 0, so it uses two ranges (Lab01: color spaces, Lab05: color tracking).
3. **Clean mask** - morphological opening removes speckles, dilation fills holes (Lab04: filtering).
4. **Combine** - `bitwise_and` with the mask and its inverse, then `cv2.add` (Lab03: bitwise ops, masking, arithmetic).
"""

    def sidebar(self, ui, mode):
        p = {"ranges": color_controls(ui, "cloak", "Red")}
        p["kernel"] = ui.slider("Mask cleanup kernel", 1, 15, 5, step=2, key="cloak_k")
        p["bg_image"] = None
        if mode == "live":
            label = "Capture background (step out!)"
        elif mode == "video":
            label = "Recapture background"
            ui.caption("In video mode the background is the median of the first 30 frames.")
        else:
            label = "Use this photo as background"
            up = ui.file_uploader("...or upload a background image", type=["jpg", "jpeg", "png"], key="cloak_bg")
            if up is not None:
                p["bg_image"] = decode_image(up.getvalue())
        p["actions"] = {"capture": action_button(ui, label, "cloak_capture")}
        return p

    def process(self, frame, p, state):
        steps = [("Original", frame)]
        if take_action(state, p, "capture"):
            if p["mode"] == "live":
                state["capture_at"] = time.time() + COUNTDOWN
                state["bg_frames"] = []
            elif p["mode"] == "video":
                state["bg_frames"] = []
                state.pop("background", None)
            else:
                state["background"] = frame.copy()

        # Live mode: countdown, then collect frames for a clean median background
        if "capture_at" in state:
            out = frame.copy()
            remaining = state["capture_at"] - time.time()
            if remaining > 0:
                put_label(out, f"Step out of the frame... {math.ceil(remaining)}", bg=(0, 0, 160))
                return out, steps
            state["bg_frames"].append(frame.copy())
            if len(state["bg_frames"]) < BG_FRAMES:
                put_label(out, f"Capturing background {len(state['bg_frames'])}/{BG_FRAMES}", bg=(0, 120, 0))
                return out, steps
            state["background"] = np.median(np.stack(state.pop("bg_frames")), axis=0).astype(np.uint8)
            del state["capture_at"]

        # Video mode: the opening frames become the background automatically
        if p["mode"] == "video" and state.get("background") is None:
            state.setdefault("bg_frames", []).append(frame.copy())
            if len(state["bg_frames"]) >= BG_FRAMES:
                state["background"] = np.median(np.stack(state.pop("bg_frames")), axis=0).astype(np.uint8)
            out = put_label(frame.copy(), "Learning the background...", bg=(0, 120, 0))
            return out, steps

        bg = p.get("bg_image")
        if bg is None:
            bg = state.get("background")
        if bg is None:
            return put_label(frame.copy(), "Capture the background first", bg=(0, 0, 160)), steps
        if bg.shape != frame.shape:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        mask = hsv_mask(frame, p["ranges"])
        clean = clean_mask(mask, p["kernel"])
        cloak_area = cv2.bitwise_and(bg, bg, mask=clean)
        rest = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(clean))
        result = cv2.add(cloak_area, rest)

        steps += [
            ("Saved background", bg),
            ("HSV mask", mask),
            ("Cleaned mask", clean),
            ("Background inside cloak", cloak_area),
            ("You, minus the cloak", rest),
        ]
        return result, steps
