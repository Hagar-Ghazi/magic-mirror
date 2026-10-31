import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool
from studio.utils import odd, put_label

KERNELS = {
    "Identity": "0 0 0\n0 1 0\n0 0 0",
    "Sharpen": "0 -1 0\n-1 5 -1\n0 -1 0",
    "Box blur": "1 1 1\n1 1 1\n1 1 1",
    "Gaussian 3x3": "1 2 1\n2 4 2\n1 2 1",
    "Laplacian edges": "0 1 0\n1 -4 1\n0 1 0",
    "Outline": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "Emboss": "-2 -1 0\n-1 1 1\n0 1 2",
    "Sobel X": "-1 0 1\n-2 0 2\n-1 0 1",
    "Sobel Y": "-1 -2 -1\n0 0 0\n1 2 1",
    "Custom": "0 -1 0\n-1 5 -1\n0 -1 0",
}


@tool("Blur and noise", "Blur", [
    P("select", "kind", "Type", "Gaussian", options=["Average (box)", "Gaussian", "Median", "Bilateral"]),
    P("slider", "ksize", "Kernel size", 21, 1, 75, 2, when=lambda v: v["kind"] != "Bilateral"),
    P("slider", "sigma", "Sigma (0 = from kernel size)", 0.0, 0.0, 30.0, 0.5, when=lambda v: v["kind"] == "Gaussian"),
    P("slider", "diameter", "Neighbourhood diameter", 9, 1, 25, 1, when=lambda v: v["kind"] == "Bilateral"),
    P("slider", "sigma_color", "Sigma color", 75, 1, 250, 1, when=lambda v: v["kind"] == "Bilateral"),
    P("slider", "sigma_space", "Sigma space", 75, 1, 250, 1, when=lambda v: v["kind"] == "Bilateral"),
], explain="""
Every blur slides a small **kernel** over the image:

- **Average** (`cv2.blur`): plain mean of the window.
- **Gaussian** (`cv2.GaussianBlur`): weighted mean, near pixels count more - the most natural blur.
- **Median** (`cv2.medianBlur`): middle value of the window - the best remedy for **salt-and-pepper** noise.
- **Bilateral** (`cv2.bilateralFilter`): only averages pixels of similar color, so it smooths skin and walls but **keeps edges**.
""")
def blur(img, kind, ksize=9, sigma=0.0, diameter=9, sigma_color=75, sigma_space=75):
    k = odd(ksize)
    if kind == "Average (box)":
        return cv2.blur(img, (k, k))
    if kind == "Gaussian":
        return cv2.GaussianBlur(img, (k, k), sigma)
    if kind == "Median":
        return cv2.medianBlur(img, k)
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


@tool("Blur and noise", "Motion blur", [
    P("slider", "length", "Length (px)", 21, 3, 101, 2),
    P("slider", "angle", "Angle (degrees)", 0, 0, 180, 5),
], explain="""
A motion-blur kernel is a single line of ones (normalized to sum 1) rotated to the direction of movement.
`cv2.filter2D` convolves it with the image, smearing every point along that line.
""")
def motion_blur(img, length, angle):
    k = odd(length)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


@tool("Blur and noise", "Add noise", [
    P("select", "kind", "Noise type", "Gaussian", options=["Gaussian", "Salt and pepper", "Speckle"]),
    P("slider", "sigma", "Strength (sigma)", 25, 1, 100, 1, when=lambda v: v["kind"] != "Salt and pepper"),
    P("slider", "amount", "Amount (fraction of pixels)", 0.05, 0.005, 0.4, 0.005, when=lambda v: v["kind"] == "Salt and pepper"),
    P("slider", "seed", "Random seed", 7, 0, 999, 1),
], explain="""
Noise is what a camera adds in low light or what a bad transmission does:

- **Gaussian**: every pixel gets a random value from a normal distribution added.
- **Salt and pepper**: a fraction of pixels become pure white or black.
- **Speckle**: multiplicative noise, `pixel + pixel * random`.

Add noise here, then try the **Blur** tool (median for salt and pepper) or **Denoise** to remove it.
""")
def add_noise(img, kind, seed, sigma=25, amount=0.05):
    rng = np.random.default_rng(int(seed))
    f = img.astype(np.float32)
    if kind == "Gaussian":
        f += rng.normal(0, sigma, img.shape)
    elif kind == "Speckle":
        f += f * rng.normal(0, sigma / 100, img.shape)
    else:
        out = img.copy()
        r = rng.random(img.shape[:2])
        out[r < amount / 2] = 0
        out[r > 1 - amount / 2] = 255
        return out
    return np.clip(f, 0, 255).astype(np.uint8)


@tool("Blur and noise", "Denoise (non-local means)", [
    P("slider", "strength", "Strength (h)", 10, 1, 40, 1),
    P("slider", "template", "Patch size", 7, 3, 15, 2),
    P("slider", "search", "Search window", 21, 7, 41, 2),
], heavy=True, explain="""
`cv2.fastNlMeansDenoisingColored` replaces each pixel by an average of pixels whose **surrounding patches look
similar**, even if they are far away. It removes noise while keeping texture and edges far better than a plain
blur, at the cost of speed.
""")
def denoise(img, strength, template, search):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, odd(template), odd(search))


@tool("Blur and noise", "Custom kernel (convolution)", [
    P("select", "preset", "Kernel", "Emboss", options=list(KERNELS)),
    P("textarea", "custom", "Kernel values (rows on lines)", KERNELS["Custom"], when=lambda v: v["preset"] == "Custom"),
    P("toggle", "normalize", "Divide by sum (keep brightness)", True),
    P("slider", "delta", "Add offset", 0, 0, 128, 1),
], explain="""
Every filter in this category is a **convolution**: each output pixel is the weighted sum of its neighbours,
with the weights given by the kernel. `cv2.filter2D` applies any kernel you type. Kernels that sum to 1 keep the
brightness, kernels that sum to 0 find edges (add an offset of 128 to see negative values).
""")
def custom_kernel(img, preset, normalize, delta, custom=KERNELS["Custom"]):
    text = custom if preset == "Custom" else KERNELS[preset]
    try:
        rows = [[float(v) for v in line.replace(",", " ").split()] for line in text.strip().splitlines() if line.strip()]
        kernel = np.array(rows, np.float32)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
            raise ValueError
    except ValueError:
        return put_label(img.copy(), "Kernel must be a square with an odd size, e.g. 3x3")
    s = kernel.sum()
    if normalize and abs(s) > 1e-6:
        kernel /= s
    return cv2.filter2D(img, -1, kernel, delta=delta)
