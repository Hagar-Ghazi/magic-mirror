# 🪞 Magic Mirror

A Streamlit app of camera "spells" built **only with classic OpenCV**: no deep learning, no pretrained models.
Every effect comes from Labs 01-05 of the ITI *Computer Vision & Pretrained Models* course
(color spaces, geometry, arithmetic and masking, filtering and edges, contours, features and detection).

## Spells

| Spell | What it does | Techniques |
|---|---|---|
| 🧥 Invisibility cloak | Pixels covered by a solid-colored cloth show the background | HSV `inRange`, morphology, `bitwise_and`, median background |
| ✍️ Air drawing | Draw in the air with a colored pen cap | Color tracking, largest contour, moments, smoothing, overlay |
| 😎 Face filters | Sunglasses, mustache and crown, plus pixelate or blur for privacy | Haar cascades, histogram equalization, alpha blending, resize |
| 🚨 Motion alarm | Boxes and an alarm when something moves | `accumulateWeighted`, `absdiff`, threshold, contours |
| 🎨 Art filters | Cartoon, pencil sketch, vintage, emboss and neon | Bilateral filter, adaptive threshold, `filter2D`, Canny |

Each spell returns its intermediate images, which the **Under the hood** tab shows step by step
with a color histogram. **How it works** explains the pipeline and shows the spell's source code.

## Sources

- **Webcam live**: real-time video through `streamlit-webrtc`
- **Camera snapshot**: one photo at a time (works on networks that block WebRTC)
- **Upload image** / **Upload video**: a processed video can be downloaded afterwards

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py               Streamlit UI: sidebar, tabs, webcam/image/video handling
core/preprocess.py   resize, mirror, decode, BGR to RGB
core/helpers.py      HSV masks, mask cleanup, contours, alpha overlay, labels
spells/base.py       Spell interface + shared sidebar controls
spells/*.py          one file per spell
```

Adding a spell means adding one file that subclasses `Spell` and registering it in `spells/__init__.py`.
