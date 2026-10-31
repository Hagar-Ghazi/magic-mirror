import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import bgr, cascade, fit_to, gray, hex_to_bgr, put_label

MATCH_METHODS = {
    "Normalized correlation (CCOEFF_NORMED)": cv2.TM_CCOEFF_NORMED,
    "Normalized cross-correlation (CCORR_NORMED)": cv2.TM_CCORR_NORMED,
    "Normalized squared difference (SQDIFF_NORMED)": cv2.TM_SQDIFF_NORMED,
}


def _line_width(img):
    return max(2, img.shape[1] // 300)


@tool("Detect", "Faces and eyes (Haar)", [
    P("slider", "scale", "Scale factor", 1.1, 1.02, 1.5, 0.01),
    P("slider", "neighbors", "Min neighbors", 5, 1, 15, 1),
    P("slider", "min_size", "Min face size (px)", 40, 10, 300, 5),
    P("toggle", "eyes", "Detect eyes too", True),
    P("toggle", "smiles", "Detect smiles", False),
    P("color", "color", "Box color", "#22C55E"),
], explain="""
Viola-Jones **Haar cascades** slide a window over the image at many scales (`scaleFactor`) and run a cascade of
simple light/dark rectangle tests; a face must be found by several overlapping windows (`minNeighbors`). Eyes and
smiles are searched only inside each face, which is faster and avoids false hits. The gray image is
**histogram-equalized** first so lighting matters less.
""")
def faces(img, scale, neighbors, min_size, eyes, smiles, color):
    g = cv2.equalizeHist(gray(img))
    found = cascade("haarcascade_frontalface_default.xml").detectMultiScale(g, scale, neighbors, minSize=(min_size, min_size))
    out, lw, c = img.copy(), _line_width(img), hex_to_bgr(color)
    for x, y, w, h in found:
        cv2.rectangle(out, (x, y), (x + w, y + h), c, lw)
        roi = g[y:y + h, x:x + w]
        if eyes:
            for ex, ey, ew, eh in cascade("haarcascade_eye.xml").detectMultiScale(roi[: h // 2 + h // 8], 1.1, 8):
                cv2.circle(out, (x + ex + ew // 2, y + ey + eh // 2), max(ew, eh) // 2, (255, 200, 0), lw, cv2.LINE_AA)
        if smiles:
            for sx, sy, sw, sh in cascade("haarcascade_smile.xml").detectMultiScale(roi[h // 2:], 1.7, 20):
                cv2.rectangle(out, (x + sx, y + h // 2 + sy), (x + sx + sw, y + h // 2 + sy + sh), (60, 60, 255), lw)
    return put_label(out, f"Faces found: {len(found)}", bg=(30, 90, 30))


@tool("Detect", "Contours and shapes", [
    P("select", "source", "Find objects with", "Otsu threshold", options=["Otsu threshold", "Canny edges", "Adaptive threshold"]),
    P("toggle", "invert", "Objects are darker than background", False),
    P("slider", "min_area", "Min area (% of image)", 0.2, 0.0, 10.0, 0.05),
    P("select", "draw", "Draw", "Contours + shape names", options=["Contours + shape names", "Contours", "Bounding boxes", "Rotated boxes", "Convex hulls", "Filled colors"]),
    P("slider", "epsilon", "Polygon simplification (%)", 3.0, 0.5, 10.0, 0.5),
    P("color", "color", "Color", "#F43F5E"),
], explain="""
1. Make a binary image (threshold or Canny).
2. `cv2.findContours` traces the outline of every white region.
3. Small contours are dropped by `cv2.contourArea`.
4. `cv2.approxPolyDP` simplifies each outline to a polygon (tolerance = % of `cv2.arcLength`); the number of
   corners names the shape: 3 = triangle, 4 = square or rectangle, many = circle.

Bounding boxes come from `cv2.boundingRect`, rotated boxes from `cv2.minAreaRect`, hulls from `cv2.convexHull`.
""")
def contours(img, source, invert, min_area, draw, epsilon, color):
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    if source == "Canny edges":
        binary = cv2.dilate(cv2.Canny(g, 50, 150), None, iterations=2)
    elif source == "Adaptive threshold":
        binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    else:
        _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if invert:
            binary = 255 - binary
    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    limit = min_area / 100 * img.shape[0] * img.shape[1]
    found = sorted([c for c in found if cv2.contourArea(c) >= limit], key=cv2.contourArea, reverse=True)
    out, lw, c = img.copy(), _line_width(img), hex_to_bgr(color)
    rng = np.random.default_rng(1)
    for cnt in found:
        if draw.startswith("Contours"):
            cv2.drawContours(out, [cnt], -1, c, lw, cv2.LINE_AA)
        elif draw == "Bounding boxes":
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(out, (x, y), (x + w, y + h), c, lw)
        elif draw == "Rotated boxes":
            cv2.drawContours(out, [np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))], -1, c, lw, cv2.LINE_AA)
        elif draw == "Convex hulls":
            cv2.drawContours(out, [cv2.convexHull(cnt)], -1, c, lw, cv2.LINE_AA)
        else:
            layer = out.copy()
            cv2.drawContours(layer, [cnt], -1, [int(v) for v in rng.integers(40, 255, 3)], -1)
            out = cv2.addWeighted(layer, 0.6, out, 0.4, 0)
        if draw == "Contours + shape names":
            approx = cv2.approxPolyDP(cnt, epsilon / 100 * cv2.arcLength(cnt, True), True)
            n = len(approx)
            if n == 3:
                name = "Triangle"
            elif n == 4:
                _, _, w, h = cv2.boundingRect(approx)
                name = "Square" if 0.9 <= w / max(h, 1) <= 1.1 else "Rectangle"
            elif n == 5:
                name = "Pentagon"
            elif n == 6:
                name = "Hexagon"
            else:
                name = "Circle"
            x, y, _, _ = cv2.boundingRect(cnt)
            put_label(out, name, org=(x, y - 4), scale=0.5, bg=(40, 40, 40))
    return put_label(out, f"Objects: {len(found)}", bg=(120, 30, 60))


@tool("Detect", "Corners (Harris, Shi-Tomasi)", [
    P("select", "method", "Method", "Shi-Tomasi", options=["Shi-Tomasi", "Harris"]),
    P("slider", "max_corners", "Max corners", 100, 5, 1000, 5, when=lambda v: v["method"] == "Shi-Tomasi"),
    P("slider", "quality", "Quality level", 0.01, 0.001, 0.3, 0.001),
    P("slider", "min_dist", "Min distance (px)", 10, 1, 100, 1, when=lambda v: v["method"] == "Shi-Tomasi"),
    P("color", "color", "Color", "#FACC15"),
], explain="""
A **corner** is a point where the image changes strongly in two directions. **Harris** (`cv2.cornerHarris`) scores
every pixel from its gradient structure; pixels above `quality * max score` are marked. **Shi-Tomasi**
(`cv2.goodFeaturesToTrack`) uses a slightly better score and returns the N strongest corners that are at least
some distance apart - the classic points to track in video.
""")
def corners(img, method, quality, color, max_corners=100, min_dist=10):
    g = gray(img)
    out, c = img.copy(), hex_to_bgr(color)
    r = max(3, img.shape[1] // 200)
    if method == "Harris":
        score = cv2.cornerHarris(np.float32(g), 2, 3, 0.04)
        mask = score > quality * score.max()
        out[cv2.dilate(mask.astype(np.uint8), None) > 0] = c
        count = int(mask.sum())
    else:
        pts = cv2.goodFeaturesToTrack(g, max_corners, quality, min_dist)
        pts = [] if pts is None else pts.reshape(-1, 2)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), r, c, -1, cv2.LINE_AA)
        count = len(pts)
    return put_label(out, f"Corners: {count}")


@tool("Detect", "ORB keypoints", [
    P("slider", "n", "Number of features", 500, 50, 3000, 50),
    P("toggle", "rich", "Show size and orientation", True),
], explain="""
**ORB** (Oriented FAST and Rotated BRIEF) finds distinctive keypoints with FAST, gives each an orientation, and
describes its neighbourhood with a 256-bit binary descriptor. Rich drawing shows each keypoint's **scale**
(circle size) and **orientation** (the line).
""")
def orb_keypoints(img, n, rich):
    orb = cv2.ORB_create(nfeatures=n)
    kps = orb.detect(gray(img), None)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0
    out = cv2.drawKeypoints(img, kps, None, (0, 255, 0), flags)
    return put_label(out, f"Keypoints: {len(kps)}")


@tool("Detect", "Feature matching (ORB)", [
    P("image", "other", "Second image (same object, other view)"),
    P("slider", "n", "Features per image", 1000, 100, 5000, 100),
    P("slider", "ratio", "Ratio test", 0.75, 0.5, 0.95, 0.01),
    P("slider", "show", "Matches to draw", 50, 5, 300, 5),
], explain="""
Keypoints and descriptors are computed with ORB in both images, then matched with a **brute-force Hamming**
matcher (`cv2.BFMatcher`). **Lowe's ratio test** keeps a match only if the best candidate is clearly better than the
second best, which removes most wrong matches. `cv2.drawMatches` puts the images side by side and joins matches.
""", heavy=True)
def feature_matching(img, other, n, ratio, show):
    if other is None:
        return put_label(img.copy(), "Upload a second image in the sidebar")
    b = bgr(other)
    orb = cv2.ORB_create(nfeatures=n)
    k1, d1 = orb.detectAndCompute(gray(img), None)
    k2, d2 = orb.detectAndCompute(gray(b), None)
    if d1 is None or d2 is None:
        return put_label(img.copy(), "No features found")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = sorted([m for m, *rest in pairs if rest and m.distance < ratio * rest[0].distance], key=lambda m: m.distance)
    out = cv2.drawMatches(img, k1, b, k2, good[:show], None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return put_label(out, f"Good matches: {len(good)}")


@tool("Detect", "Template matching", [
    P("select", "source", "Template", "Region of this image", options=["Region of this image", "Upload a template"]),
    P("range", "tx", "Template X (%)", (40, 60), 0, 100, 1, when=lambda v: v["source"] == "Region of this image"),
    P("range", "ty", "Template Y (%)", (40, 60), 0, 100, 1, when=lambda v: v["source"] == "Region of this image"),
    P("image", "other", "Template image", when=lambda v: v["source"] == "Upload a template"),
    P("select", "method", "Method", list(MATCH_METHODS)[0], options=list(MATCH_METHODS)),
    P("slider", "threshold", "Match threshold", 0.8, 0.3, 1.0, 0.01),
    P("toggle", "show_map", "Show the score map", False),
], explain="""
`cv2.matchTemplate` slides the template over the image and computes a similarity score at every position, giving a
**score map** (bright = good match). Every position above the threshold is a match; overlapping hits are merged
with non-maximum suppression. It only works when the object has the same size and rotation as the template.
""")
def template_matching(img, source, method, threshold, show_map, tx=(40, 60), ty=(40, 60), other=None):
    g = gray(img)
    h, w = g.shape
    if source == "Upload a template":
        if other is None:
            return put_label(img.copy(), "Upload a template image in the sidebar")
        t = gray(bgr(other))
        if t.shape[0] > h or t.shape[1] > w:
            t = cv2.resize(t, (min(t.shape[1], w), min(t.shape[0], h)))
    else:
        x1, x2 = int(tx[0] / 100 * w), max(int(tx[1] / 100 * w), int(tx[0] / 100 * w) + 4)
        y1, y2 = int(ty[0] / 100 * h), max(int(ty[1] / 100 * h), int(ty[0] / 100 * h) + 4)
        t = g[y1:y2, x1:x2]
    m = MATCH_METHODS[method]
    scores = cv2.matchTemplate(g, t, m)
    if m == cv2.TM_SQDIFF_NORMED:
        scores = 1 - scores
    if show_map:
        return bgr(cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    th, tw = t.shape
    ys, xs = np.where(scores >= threshold)
    boxes = [[int(x), int(y), tw, th] for x, y in zip(xs, ys)]
    keep = cv2.dnn.NMSBoxes(boxes, [float(scores[y, x]) for x, y in zip(xs, ys)], threshold, 0.3) if boxes else []
    out, lw = img.copy(), _line_width(img)
    for i in np.array(keep).reshape(-1):
        x, y, bw, bh = boxes[i]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), lw)
    if source == "Region of this image":
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), max(1, lw // 2))
    return put_label(out, f"Matches: {len(np.array(keep).reshape(-1))}  best score: {scores.max():.2f}")


@tool("Detect", "Document scanner", [
    P("slider", "blur", "Blur before edges", 5, 1, 15, 2),
    P("range", "canny", "Edge thresholds", (50, 150), 0, 400, 5),
    P("select", "finish", "Finish", "Color", options=["Color", "Gray", "Black and white (adaptive)"]),
], explain="""
1. Blur and **Canny** to find edges, dilate to close gaps.
2. `cv2.findContours` and keep the largest contour that `cv2.approxPolyDP` simplifies to **4 corners** (the page).
3. Order the corners, compute the page size, and `cv2.getPerspectiveTransform` + `cv2.warpPerspective` flatten it
   to a straight, top-down view.
4. Optionally an adaptive threshold for a clean black-and-white scan.
""")
def document(img, blur, canny, finish):
    g = cv2.GaussianBlur(gray(img), (blur | 1, blur | 1), 0)
    edges = cv2.dilate(cv2.Canny(g, *canny), None, iterations=2)
    found, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = None
    for cnt in sorted(found, key=cv2.contourArea, reverse=True)[:8]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * img.shape[0] * img.shape[1]:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return put_label(img.copy(), "No 4-corner page found - adjust the edge thresholds")
    s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
    tl, br, tr, bl = quad[np.argmin(s)], quad[np.argmax(s)], quad[np.argmin(d)], quad[np.argmax(d)]
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    m = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    page = cv2.warpPerspective(img, m, (w, h))
    if finish == "Gray":
        return bgr(gray(page))
    if finish.startswith("Black"):
        return bgr(cv2.adaptiveThreshold(gray(page), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10))
    return page
