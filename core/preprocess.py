import cv2


def prepare(frame, max_width=640, flip=True):
    """Resize (keeping aspect ratio) and optionally mirror a BGR frame."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def decode_image(data):
    """Decode uploaded image bytes into a BGR array (or None)."""
    import numpy as np

    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_rgb(img):
    """Convert BGR or grayscale image to RGB for display in Streamlit."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
