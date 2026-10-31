import cv2
import numpy as np

from studio.registry import Param as P
from studio.registry import tool


@tool("Adjust", "Brightness and contrast", [
    P("slider", "contrast", "Contrast (alpha)", 1.3, 0.0, 3.0, 0.05),
    P("slider", "brightness", "Brightness (beta)", 10, -127, 127, 1),
], explain="""
`out = alpha * pixel + beta`, then clipped to 0-255 by `cv2.convertScaleAbs`.
Alpha above 1 stretches the difference between dark and light pixels (more contrast); beta shifts every
pixel up or down (brighter or darker). Clipping is why very bright areas turn flat white.
""")
def brightness_contrast(img, contrast, brightness):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


@tool("Adjust", "Gamma correction", [
    P("slider", "gamma", "Gamma", 1.8, 0.1, 3.0, 0.05, help="Below 1 darkens, above 1 brightens shadows"),
], explain="""
Each pixel becomes `255 * (pixel / 255) ** (1 / gamma)`. Unlike brightness, gamma lifts or
darkens the mid-tones while black stays black and white stays white. A lookup table (`cv2.LUT`) applies it fast.
""")
def gamma(img, gamma):
    table = (255 * (np.arange(256) / 255.0) ** (1.0 / gamma)).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


@tool("Adjust", "Hue, saturation and value", [
    P("slider", "hue", "Hue shift (degrees)", 40, -180, 180, 2),
    P("slider", "saturation", "Saturation", 1.5, 0.0, 3.0, 0.05),
    P("slider", "value", "Value (lightness)", 1.0, 0.0, 2.0, 0.05),
], explain="""
The image is converted to **HSV**. Hue is an angle on the color wheel (OpenCV stores 0-179, so one unit is
2 degrees), saturation is color purity and value is lightness. Shifting hue rotates every color,
scaling saturation makes colors richer or grayer.
""")
def hsv_adjust(img, hue, saturation, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue / 2) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


@tool("Adjust", "Temperature and tint", [
    P("slider", "temperature", "Temperature (cool to warm)", -60, -100, 100, 1),
    P("slider", "tint", "Tint (green to magenta)", 0, -100, 100, 1),
], explain="""
Warm adds red and removes blue; cool does the opposite. Tint moves the green channel against red and blue.
This works directly on the B, G, R channels with saturating `cv2.add` / `cv2.subtract`.
""")
def temperature(img, temperature, tint):
    b, g, r = cv2.split(img.astype(np.int16))
    t = temperature * 0.6
    r, b = r + t, b - t
    g = g - tint * 0.5
    r, b = r + tint * 0.25, b + tint * 0.25
    return cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in (b, g, r)])


@tool("Adjust", "Sharpen", [
    P("slider", "amount", "Amount", 2.5, 0.0, 4.0, 0.1),
    P("slider", "radius", "Radius (sigma)", 3.0, 0.5, 10.0, 0.5),
], explain="""
**Unsharp masking:** blur the image, then add back the difference between the original and the blur:
`out = (1 + amount) * img - amount * blurred` with `cv2.addWeighted`. Edges get stronger, flat areas stay the same.
""")
def sharpen(img, amount, radius):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


@tool("Adjust", "Invert (negative)", [], explain="""
Every pixel becomes `255 - pixel` using `cv2.bitwise_not`, like a photographic negative.
""")
def invert(img):
    return cv2.bitwise_not(img)


@tool("Adjust", "Posterize", [
    P("slider", "levels", "Levels per channel", 4, 2, 16, 1),
], explain="""
Reduces each channel to a few levels by integer division: `(pixel // step) * step`. Fewer levels give a
flat, poster-like look and show what **quantization** (bit depth) does to an image.
""")
def posterize(img, levels):
    step = 256 // levels
    return ((img // step) * step + step // 2).clip(0, 255).astype(np.uint8)
