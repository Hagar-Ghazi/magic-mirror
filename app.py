import inspect
import os
import tempfile
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from core.helpers import put_label
from core.preprocess import decode_image, prepare, to_rgb
from spells import ALL_SPELLS

st.set_page_config(page_title="Magic Mirror", page_icon="🪞", layout="wide")

SPELLS = {f"{s.icon} {s.name}": s for s in ALL_SPELLS}
SOURCES = {"Webcam live": "live", "Camera snapshot": "image", "Upload image": "image", "Upload video": "video"}

# streamlit-webrtc picks a TURN server from these env vars (needed on Streamlit Cloud,
# where STUN alone usually cannot connect). Copy them from st.secrets if set there.
TURN_KEYS = ["HF_TOKEN", "CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN",
             "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]
try:
    for k in TURN_KEYS:
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:  # no secrets file locally
    pass
HAS_TURN = any(os.getenv(k) for k in TURN_KEYS)


class MirrorProcessor(VideoProcessorBase):
    """Runs the selected spell on every webcam frame (in streamlit-webrtc's worker thread)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.spell, self.params = None, {}
        self.flip, self.max_width, self.show_fps = True, 640, True
        self.states = {}
        self.latest_result, self.latest_steps = None, []
        self.times = deque(maxlen=30)

    def update(self, spell, params, flip, max_width, show_fps):
        with self.lock:
            self.spell, self.params = spell, params
            self.flip, self.max_width, self.show_fps = flip, max_width, show_fps

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            spell, params = self.spell, self.params
            flip, max_width, show_fps = self.flip, self.max_width, self.show_fps
        img = prepare(img, max_width, flip)
        result, steps = img, [("Original", img)]
        if spell is not None:
            try:
                result, steps = spell.process(img, params, self.states.setdefault(spell.name, {}))
            except Exception as exc:  # keep the stream alive if a spell fails
                result = put_label(img.copy(), f"Error: {exc}"[:60], bg=(0, 0, 160), scale=0.5)
        self.times.append(time.time())
        if show_fps and len(self.times) > 1:
            fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
            put_label(result, f"{fps:.0f} FPS", org=(result.shape[1] - 95, 30), scale=0.6)
        self.latest_result, self.latest_steps = result, steps
        return av.VideoFrame.from_ndarray(result, format="bgr24")


def spell_state(spell):
    states = st.session_state.setdefault("spell_states", {})
    return states.setdefault(spell.name, {})


def save_to_gallery(img, caption):
    st.session_state.setdefault("gallery", []).insert(0, (caption, img.copy()))
    st.toast("Saved to gallery")


def show_steps(steps, result):
    if not steps:
        st.info("Run a spell first - the intermediate steps will appear here.")
        return
    cols = st.columns(3)
    for i, (title, img) in enumerate(steps):
        cols[i % 3].image(to_rgb(img), caption=f"{i + 1}. {title}", width="stretch")
    if result is not None:
        st.markdown("**Color histogram of the result**")
        hist = {
            name: cv2.calcHist([result], [ch], None, [64], [0, 256]).ravel()
            for ch, name in enumerate(["Blue", "Green", "Red"])
        }
        st.line_chart(pd.DataFrame(hist), color=["#3B82F6", "#22C55E", "#EF4444"], height=220)


def open_writer(shape, fps):
    """Pick the first codec this OpenCV build can write, preferring browser-playable ones."""
    h, w = shape[:2]
    for codec, ext, mime in [("avc1", ".mp4", "video/mp4"), ("VP80", ".webm", "video/webm"), ("mp4v", ".mp4", "video/mp4")]:
        path = os.path.join(tempfile.gettempdir(), f"magic_mirror_out{ext}")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            return writer, path, mime
    raise RuntimeError("No video codec available")


def process_video(data, spell, params, flip, max_width, max_frames):
    src = os.path.join(tempfile.gettempdir(), "magic_mirror_in.mp4")
    with open(src, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or max_frames, max_frames))
    state, writer, mid = {}, None, None
    bar = st.progress(0.0, text="Casting the spell on every frame...")
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frame = prepare(frame, max_width, flip)
        result, steps = spell.process(frame, params, state)
        if writer is None:
            writer, out_path, mime = open_writer(result.shape, fps)
        writer.write(result)
        if i == total // 2 or mid is None:
            mid = (result, steps)
        bar.progress((i + 1) / total, text=f"Frame {i + 1}/{total}")
    cap.release()
    bar.empty()
    if writer is None:
        st.error("Could not read any frames from this video.")
        return None
    writer.release()
    with open(out_path, "rb") as f:
        return {"video": f.read(), "mime": mime, "result": mid[0], "steps": mid[1]}


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.title("🪞 Magic Mirror")
sb.caption("Classic OpenCV spells - no deep learning.")
source = sb.radio("Source", list(SOURCES), key="source")
mode = SOURCES[source]
spell_label = sb.selectbox("Spell", list(SPELLS), key="spell")
spell = SPELLS[spell_label]
sb.caption(spell.tagline)

sb.subheader("Spell settings")
params = spell.sidebar(sb, mode)
params["mode"] = mode

sb.subheader("Display")
flip = sb.toggle("Mirror flip", mode == "live", key="flip")
show_fps = sb.toggle("Show FPS", True, key="fps") if mode == "live" else False
max_width = sb.select_slider("Processing width", [320, 480, 640, 800, 1024], 640, key="width")
if sb.button("Reset this spell", width="stretch"):
    st.session_state.get("spell_states", {}).pop(spell.name, None)
    st.session_state["video_out"] = None

# ---------------- Main ----------------
st.title(f"{spell.icon} {spell.name}")
tab_live, tab_hood, tab_gallery, tab_how = st.tabs(["Live", "Under the hood", "Gallery", "How it works"])
result, steps = None, []

with tab_live:
    if mode == "live":
        ctx = webrtc_streamer(
            key="mirror",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MirrorProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        processor = ctx.video_processor
        if processor:
            processor.update(spell, params, flip, max_width, show_fps)
            result, steps = processor.latest_result, processor.latest_steps
            if st.button("📸 Snapshot to gallery") and result is not None:
                save_to_gallery(result, spell.name)
        else:
            st.info("Press **START** and allow camera access. If the video never connects "
                    "(some networks block WebRTC), switch the source to **Camera snapshot**.")
            if not HAS_TURN:
                st.caption("No TURN server configured, so live video may not connect on the cloud. "
                           "Add `HF_TOKEN` to the app secrets (see README).")
    elif mode == "image":
        if source == "Camera snapshot":
            shot = st.camera_input("Take a photo")
        else:
            shot = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if shot is not None:
            frame = decode_image(shot.getvalue())
            if frame is None:
                st.error("That file is not an image OpenCV can read.")
            else:
                frame = prepare(frame, max_width, flip)
                result, steps = spell.process(frame, params, spell_state(spell))
                st.image(to_rgb(result), width="stretch")
                c1, c2 = st.columns(2)
                if c1.button("📸 Save to gallery", width="stretch"):
                    save_to_gallery(result, spell.name)
                c2.download_button("⬇️ Download", cv2.imencode(".png", result)[1].tobytes(),
                                   "magic_mirror.png", "image/png", width="stretch")
    else:
        up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
        max_frames = st.slider("Max frames to process", 30, 900, 300, step=30)
        if up is not None and st.button("✨ Cast spell on video", type="primary"):
            st.session_state["video_out"] = process_video(up.getvalue(), spell, params, flip, max_width, max_frames)
        out = st.session_state.get("video_out")
        if out:
            st.video(out["video"], format=out["mime"])
            st.download_button("⬇️ Download result", out["video"],
                               "magic_mirror" + (".webm" if out["mime"] == "video/webm" else ".mp4"), out["mime"])
            result, steps = out["result"], out["steps"]

with tab_hood:
    st.caption("Every step the spell took on the current frame (middle frame for videos).")
    if mode == "live":
        st.button("🔄 Refresh from live video")
    show_steps(steps, result)

with tab_gallery:
    gallery = st.session_state.get("gallery", [])
    if not gallery:
        st.info("Snapshots you save will appear here.")
    cols = st.columns(3)
    for i, (caption, img) in enumerate(gallery):
        with cols[i % 3]:
            st.image(to_rgb(img), caption=caption, width="stretch")
            st.download_button("Download", cv2.imencode(".png", img)[1].tobytes(), f"snapshot_{i}.png",
                               "image/png", key=f"dl_{i}")

with tab_how:
    st.markdown(spell.explain)
    with st.expander("Show the code of this spell"):
        st.code(inspect.getsource(type(spell).process), language="python")
    st.markdown("---")
    st.markdown(
        "**Built with:** Streamlit, streamlit-webrtc, OpenCV, NumPy. "
        "Every effect uses only classic computer vision from Labs 01-05."
    )
