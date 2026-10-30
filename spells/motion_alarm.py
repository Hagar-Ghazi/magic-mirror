import cv2

from core.helpers import put_label
from spells.base import Spell, action_button, take_action


class MotionAlarm(Spell):
    name = "Motion alarm"
    icon = "🚨"
    tagline = "A tiny security camera: boxes around anything that moves."
    explain = """
**Idea:** keep a slowly-updating picture of the empty scene and flag pixels that differ from it.

1. **Grayscale + Gaussian blur** to ignore sensor noise (Lab04).
2. **Running average** `cv2.accumulateWeighted` is the background model, so slow light changes do not trigger it.
3. **Difference** `cv2.absdiff` between the frame and the model (Lab03 arithmetic, Lab05 motion detection).
4. **Threshold + dilate**, then **contours** bigger than the minimum area become alarm boxes (Lab05).
"""

    def sidebar(self, ui, mode):
        if mode == "image":
            ui.info("Motion needs several frames: use webcam or video mode.")
        p = {}
        p["threshold"] = ui.slider("Difference threshold", 5, 100, 25, key="motion_t")
        p["min_area"] = ui.slider("Min moving area (px²)", 100, 10000, 1500, step=100, key="motion_area")
        p["adapt"] = ui.slider("Background adapt speed", 0.01, 0.5, 0.05, step=0.01, key="motion_alpha")
        p["actions"] = {"reset": action_button(ui, "Reset background", "motion_reset")}
        return p

    def process(self, frame, p, state):
        if take_action(state, p, "reset"):
            state.pop("avg", None)
            state["events"] = 0

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if state.get("avg") is None or state["avg"].shape != gray.shape:
            state["avg"] = gray.astype("float")
            return put_label(frame.copy(), "Learning the scene...", bg=(0, 120, 0)), [("Original", frame)]

        cv2.accumulateWeighted(gray, state["avg"], p["adapt"])
        model = cv2.convertScaleAbs(state["avg"])
        diff = cv2.absdiff(gray, model)
        _, thresh = cv2.threshold(diff, p["threshold"], 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy()
        moving = [c for c in contours if cv2.contourArea(c) >= p["min_area"]]
        for c in moving:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        alarm = bool(moving)
        if alarm and not state.get("alarm"):
            state["events"] = state.get("events", 0) + 1
        state["alarm"] = alarm

        if alarm:
            cv2.rectangle(result, (0, 0), (result.shape[1] - 1, result.shape[0] - 1), (0, 0, 255), 8)
            put_label(result, "MOTION DETECTED", bg=(0, 0, 200))
        else:
            put_label(result, "All quiet", bg=(0, 120, 0))
        put_label(result, f"Alarms: {state.get('events', 0)}", org=(10, result.shape[0] - 15), scale=0.6)

        steps = [
            ("Blurred grayscale", gray),
            ("Background model", model),
            ("Difference", diff),
            ("Threshold + dilate", thresh),
        ]
        return result, steps
