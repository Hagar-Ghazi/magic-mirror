import hashlib
import inspect
import os
from math import gcd

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import studio.ops  # noqa: F401  (registers every tool)
from studio.registry import CATEGORIES, TOOLS, by_category, defaults, describe, run
from studio.utils import ASSETS, bgr, emoji_names, load_emoji

st.set_page_config(page_title="Photo Studio", page_icon="🎨", layout="wide")

MAX_HISTORY = 40
SIZES = [640, 960, 1280, 1600, 2048]
ss = st.session_state


# ---------------- helpers ----------------
def to_rgb(img):
    return cv2.cvtColor(bgr(img), cv2.COLOR_BGR2RGB)


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:  # 16-bit PNG/TIFF
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, img.max()))
    return bgr(img)


def limit_size(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def history_sig():
    return repr([(s["tool"], sorted(s["params"].items())) for s in ss.history])


def rebuild(start=0):
    """Replay history from step `start` (cache[i] is the image after i steps)."""
    ss.cache = ss.cache[:start + 1]
    for step in ss.history[start:]:
        try:
            ss.cache.append(run(TOOLS[step["tool"]], ss.cache[-1], step["params"], ss.aux))
        except Exception as exc:  # keep going so one bad step cannot lock the editor
            st.toast(f"{step['tool']} failed: {exc}")
            ss.cache.append(ss.cache[-1])


def load_original(img, key):
    ss.source_key = key
    ss.original_full = img
    ss.history, ss.redo, ss.preview = [], [], {}
    ss.cache = [limit_size(img, ss.get("max_side", 1280))]


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def jpg_bytes(img, q):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def widget(ui, tool, p):
    key = f"{tool.name}|{p.key}"
    if p.kind == "slider":
        return ui.slider(p.label, p.min, p.max, p.default, p.step, key=key, help=p.help)
    if p.kind == "range":
        return tuple(ui.slider(p.label, p.min, p.max, tuple(p.default), p.step, key=key, help=p.help))
    if p.kind == "select":
        return ui.selectbox(p.label, p.options, index=p.options.index(p.default), key=key, help=p.help)
    if p.kind == "toggle":
        return ui.toggle(p.label, p.default, key=key, help=p.help)
    if p.kind == "color":
        return ui.color_picker(p.label, p.default, key=key, help=p.help)
    if p.kind == "text":
        return ui.text_input(p.label, p.default, key=key, help=p.help)
    if p.kind == "textarea":
        return ui.text_area(p.label, p.default, key=key, help=p.help, height=100)
    if p.kind == "emoji":
        names = emoji_names()
        c1, c2 = ui.columns([4, 1], vertical_alignment="bottom")
        choice = c1.selectbox(p.label, names, index=names.index(p.default) if p.default in names else 0, key=key)
        icon = load_emoji(choice)
        if icon is not None:
            c2.image(cv2.cvtColor(icon, cv2.COLOR_BGRA2RGBA), width=40)
        return choice
    if p.kind == "image":
        up = ui.file_uploader(p.label, type=["jpg", "jpeg", "png", "webp", "bmp"], key=key)
        if up is None:
            return None
        digest = hashlib.sha1(up.getvalue()).hexdigest()[:12]
        if digest not in ss.aux:
            img = decode(up.getvalue())
            if img is None:
                ui.error("Could not read that image.")
                return None
            ss.aux[digest] = limit_size(img, 1600)
        return digest
    raise ValueError(p.kind)


def histogram_frame(img):
    data = {n: cv2.calcHist([img], [i], None, [128], [0, 256]).ravel() for i, n in enumerate(["Blue", "Green", "Red"])}
    data["Gray"] = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [128], [0, 256]).ravel()
    df = pd.DataFrame(data)
    df.index = df.index * 2
    return df


def show(where, img, label, max_height=620):
    """Display an image without letting tall (portrait) photos push everything off screen."""
    h, w = img.shape[:2]
    width = int(max_height * w / h)
    where.image(to_rgb(img), caption=f"{label} - {w} x {h} px", width=width if width < 900 else "stretch")


# ---------------- state ----------------
ss.setdefault("aux", {})
ss.setdefault("history", [])
ss.setdefault("redo", [])
ss.setdefault("preview", {})

# ---------------- sidebar: image ----------------
sb = st.sidebar
sb.title("🎨 Photo Studio")
sb.caption("Every classic image operation from the course, in one editor.")
upload = sb.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])
if upload is not None:
    key = f"upload:{upload.name}:{upload.size}"
    if ss.get("source_key") != key:
        img = decode(upload.getvalue())
        if img is None:
            sb.error("That file is not an image OpenCV can read.")
        else:
            load_original(img, key)
elif ss.get("source_key") != "sample":
    load_original(cv2.imread(os.path.join(ASSETS, "sample.jpg")), "sample")
if ss.source_key == "sample":
    sb.info("Showing a sample photo. Upload your own to edit it.")

max_side = sb.select_slider("Working size (longest side, px)", SIZES, value=ss.get("max_side", 1280),
                            help="Big photos are scaled down while editing so every tool stays fast.")
if max_side != ss.get("max_side", 1280):
    ss.max_side = max_side
    ss.cache = [limit_size(ss.original_full, max_side)]
    rebuild(0)

# ---------------- sidebar: tool ----------------
sb.divider()
category = sb.pills("Category", CATEGORIES, default="Adjust", key="category") or "Adjust"
tools = by_category(category)
tool_name = sb.selectbox("Tool", [t.name for t in tools], key=f"tool_{category}")
tool = TOOLS[tool_name]

params = {}
for p in tool.params:
    if p.when and not p.when({**defaults(tool), **params}):
        params[p.key] = p.default
        continue
    params[p.key] = widget(sb, tool, p)

live = sb.toggle("Live preview", not tool.heavy, key=f"live|{tool.name}",
                 help="Slow tools preview only when you press Preview.")
c1, c2 = sb.columns(2)
apply_clicked = c1.button("Apply", type="primary", width="stretch")
if c2.button("Reset settings", width="stretch"):
    for p in tool.params:
        ss.pop(f"{tool.name}|{p.key}", None)
    st.rerun()
preview_clicked = False if live else sb.button("Preview", width="stretch")

# ---------------- compute preview ----------------
current = ss.cache[-1]
pkey = (history_sig(), ss.get("max_side", 1280), tool.name, repr(sorted(params.items())))
after, error = None, None
just_applied = ss.get("applied_key") == pkey  # don't preview the same tool twice right after Apply
if (live and not just_applied) or preview_clicked or apply_clicked:
    if pkey in ss.preview:
        after = ss.preview[pkey]
    else:
        try:
            with st.spinner(f"Applying {tool.name}..."):
                after = run(tool, current, params, ss.aux)
            ss.preview = {pkey: after}
        except Exception as exc:
            error = exc

if apply_clicked and after is not None:
    ss.history.append({"tool": tool.name, "params": params})
    ss.cache.append(after)
    ss.redo = []
    if len(ss.history) > MAX_HISTORY:  # fold the oldest step into the base image
        ss.history.pop(0)
        ss.cache.pop(0)
        ss.original_full = ss.cache[0]
    st.toast(f"Applied {tool.name}")
    current = ss.cache[-1]
    ss.applied_key = (history_sig(), ss.get("max_side", 1280), tool.name, repr(sorted(params.items())))
    just_applied = True

# ---------------- main: toolbar ----------------
st.markdown(f"### {tool.name}  \n<span style='opacity:.7'>{category}</span>", unsafe_allow_html=True)
bar = st.columns([1, 1, 1, 1.4, 1.4, 2.4])
if bar[0].button("Undo", disabled=not ss.history, width="stretch"):
    ss.redo.append(ss.history.pop())
    ss.cache.pop()
    st.rerun()
if bar[1].button("Redo", disabled=not ss.redo, width="stretch"):
    step = ss.redo.pop()
    ss.history.append(step)
    ss.cache.append(run(TOOLS[step["tool"]], ss.cache[-1], step["params"], ss.aux))
    st.rerun()
if bar[2].button("Reset all", disabled=not ss.history, width="stretch"):
    ss.redo = list(reversed(ss.history))
    ss.history = []
    ss.cache = ss.cache[:1]
    st.rerun()
shown_after = after if after is not None else current
bar[3].download_button("Download PNG", png_bytes(shown_after), "photo_studio.png", "image/png", width="stretch")
bar[4].download_button("Download JPG", jpg_bytes(shown_after, 92), "photo_studio.jpg", "image/jpeg", width="stretch")
view = bar[5].segmented_control("View", ["Side by side", "After", "Before"], default="Side by side",
                                label_visibility="collapsed") or "Side by side"
compare_to = st.radio("Before means", ["Before this tool", "Original photo"], horizontal=True,
                      label_visibility="collapsed")

if error is not None:
    st.error(f"{tool.name} could not run with these settings: {error}")
before = current if compare_to == "Before this tool" else ss.cache[0]
if apply_clicked:
    before = ss.cache[-2] if compare_to == "Before this tool" and len(ss.cache) > 1 else before

if view == "Side by side":
    left, right = st.columns(2)
    show(left, before, "Before")
    show(right, shown_after, "After")
else:
    show(st, shown_after if view == "After" else before, view, max_height=760)
if just_applied:
    st.success(f"{tool.name} applied. Change a setting to preview it again on top, or pick another tool.")
elif not live and after is None:
    st.info("Live preview is off for this tool. Press **Preview** or **Apply** in the sidebar.")

# ---------------- main: panels ----------------
tab_how, tab_hist, tab_pixel, tab_history, tab_info = st.tabs(
    ["How it works", "Histogram", "Pixel inspector", f"History ({len(ss.history)})", "Image info"])

with tab_how:
    st.markdown(tool.explain or "_No description._")
    with st.expander("Show the code of this tool"):
        st.code(inspect.getsource(tool.fn), language="python")

with tab_hist:
    hc1, hc2 = st.columns(2)
    colors = ["#3B82F6", "#22C55E", "#EF4444", "#A3A3A3"]
    hc1.markdown("**Before**")
    hc1.line_chart(histogram_frame(before), color=colors, height=260)
    hc2.markdown("**After**")
    hc2.line_chart(histogram_frame(shown_after), color=colors, height=260)
    g1, g2 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(shown_after, cv2.COLOR_BGR2GRAY)
    m = st.columns(4)
    m[0].metric("Mean brightness", f"{g2.mean():.0f}", f"{g2.mean() - g1.mean():+.0f}")
    m[1].metric("Contrast (std)", f"{g2.std():.0f}", f"{g2.std() - g1.std():+.0f}")
    m[2].metric("Darkest", int(g2.min()))
    m[3].metric("Brightest", int(g2.max()))
    st.caption("x axis: pixel value 0-255, y axis: number of pixels. Try Histogram equalization or "
               "Brightness and contrast and watch the curves move.")

with tab_pixel:
    img = shown_after
    h, w = img.shape[:2]
    pc1, pc2 = st.columns(2)
    px = pc1.slider("X (column)", 0, w - 1, w // 2)
    py = pc1.slider("Y (row)", 0, h - 1, h // 2)
    b, g, r = (int(v) for v in img[py, px])
    one = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(one, cv2.COLOR_BGR2HSV)[0, 0]
    lab = cv2.cvtColor(one, cv2.COLOR_BGR2LAB)[0, 0]
    gray_v = int(cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)[0, 0])
    pc1.markdown(
        f"| Space | Values |\n|---|---|\n"
        f"| BGR (OpenCV order) | {b}, {g}, {r} |\n| RGB | {r}, {g}, {b} |\n"
        f"| HSV | {hsv[0]}, {hsv[1]}, {hsv[2]} |\n| LAB | {lab[0]}, {lab[1]}, {lab[2]} |\n"
        f"| Gray | {gray_v} |\n| Hex | #{r:02X}{g:02X}{b:02X} |\n\n"
        f"In NumPy this pixel is `img[{py}, {px}]`: **row first, then column**."
    )
    k = 7
    patch = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_CONSTANT, value=0)[py:py + 2 * k + 1, px:px + 2 * k + 1]
    zoom = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_NEAREST)
    c = 300 // (2 * k + 1)
    cv2.rectangle(zoom, (k * c, k * c), ((k + 1) * c, (k + 1) * c), (0, 255, 255), 2)
    pc2.image(to_rgb(zoom), caption="15 x 15 pixels around the point (nearest-neighbour zoom)", width=300)

with tab_history:
    if not ss.history:
        st.info("No edits yet. Choose a tool in the sidebar and press Apply.")
    for i, step in enumerate(ss.history):
        hc = st.columns([0.5, 5, 1.2, 1.4])
        hc[0].markdown(f"**{i + 1}**")
        hc[1].markdown(f"**{step['tool']}**  \n<span style='opacity:.7'>{describe(TOOLS[step['tool']], step['params'])}</span>",
                       unsafe_allow_html=True)
        if hc[2].button("Remove", key=f"rm{i}", width="stretch"):
            ss.history.pop(i)
            rebuild(i)
            st.rerun()
        if hc[3].button("Back to here", key=f"bk{i}", width="stretch"):
            ss.redo = list(reversed(ss.history[i + 1:]))
            ss.history = ss.history[:i + 1]
            ss.cache = ss.cache[:i + 2]
            st.rerun()

with tab_info:
    img = shown_after
    h, w = img.shape[:2]
    d = gcd(w, h)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Width x height", f"{w} x {h}")
    i2.metric("Aspect ratio", f"{w // d}:{h // d}" if w // d < 50 else f"{w / h:.2f}:1")
    i3.metric("Pixels", f"{w * h / 1e6:.2f} MP")
    i4.metric("Shape (NumPy)", str(img.shape))
    j1, j2, j3, j4 = st.columns(4)
    mean = img.reshape(-1, 3).mean(0)
    j1.metric("Data type", str(img.dtype))
    j2.metric("Mean B, G, R", ", ".join(f"{v:.0f}" for v in mean))
    j3.metric("PNG size", f"{len(png_bytes(img)) / 1024:.0f} KB")
    j4.metric("JPG size (q=92)", f"{len(jpg_bytes(img, 92)) / 1024:.0f} KB")
    oh, ow = ss.original_full.shape[:2]
    st.caption(f"Original upload: {ow} x {oh} px. Editing at up to {ss.get('max_side', 1280)} px on the longest side.")
