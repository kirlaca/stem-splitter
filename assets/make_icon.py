"""Generates assets/icon.png and assets/icon.ico (run once; output is committed)."""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg = Image.new("RGBA", (S, S))
g = ImageDraw.Draw(bg)
for y in range(S):  # vertical gradient
    t = y / S
    g.line([(0, y), (S, y)], fill=(int(32 + 20 * t), int(40 + 10 * t), int(70 + 40 * t), 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([16, 16, S - 16, S - 16], radius=110, fill=255)
img.paste(bg, (0, 0), mask)
d = ImageDraw.Draw(img)
colors = [(96, 205, 255), (108, 203, 95), (252, 225, 0), (255, 140, 90), (220, 120, 255)]
heights = [0.55, 0.85, 0.65, 0.95, 0.5]
n, bw, gap = len(colors), 54, 26
x0 = (S - (n * bw + (n - 1) * gap)) // 2
for i, (c, h) in enumerate(zip(colors, heights)):
    x = x0 + i * (bw + gap)
    hh = int(300 * h)
    d.rounded_rectangle([x, S // 2 - hh // 2, x + bw, S // 2 + hh // 2], radius=bw // 2, fill=c + (255,))
img.save(os.path.join(HERE, "icon.png"))
img.save(os.path.join(HERE, "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("ok")
