"""Draw each Goal Force input (force arrow + mass blobs) on its first frame, exactly as
Goal Force will see it (image stretched to 832x480, y measured from the bottom).
Writes inputs/previews/<name>.png. Run from the repo root after make_goalforce_csvs.py."""
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

IN = Path(__file__).resolve().parent / "inputs"
W, H = 832, 480
MIN_F, MAX_F, MIN_M, MAX_M = 30.0, 400.0, 1.0, 4.0     # hard-coded in inference_goal_force.py


def main():
    out = IN / "previews"
    out.mkdir(exist_ok=True)
    for csv in sorted(IN.glob("*.csv")):
        if csv.name == "manifest.csv":
            continue
        r = pd.read_csv(csv).iloc[0]
        im = Image.open(IN / "images" / r["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        if r.projectile_force_magnitude > -1:
            x0 = r.projectile_coordx / r.width * W
            y0 = (1 - r.projectile_coordy / r.height) * H
            t = (r.projectile_force_magnitude - MIN_F) / (MAX_F - MIN_F)
            L = W / 8 + (W / 2 - W / 8) * t
            a = math.radians(r.projectile_force_angle)
            x1, y1 = x0 + L * math.cos(a), y0 - L * math.sin(a)
            d.line([x0, y0, x1, y1], fill=(255, 0, 0), width=4)
            d.ellipse([x1 - 7, y1 - 7, x1 + 7, y1 + 7], fill=(255, 0, 0))
            d.ellipse([x0 - 20, y0 - 20, x0 + 20, y0 + 20], outline=(255, 0, 0), width=2)
        for slot, col in (("projectile", (0, 120, 255)), ("target", (0, 200, 0))):
            m = r[f"{slot}_mass"]
            if m > -1:
                rad = 5 + 35 * (m - MIN_M) / (MAX_M - MIN_M)
                x, y = r[f"{slot}_coordx"], H - r[f"{slot}_coordy"]
                d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=col, width=3)
                d.text((x + rad + 3, y - 6), f"m={m:g}", fill=col)
        d.text((6, 6), csv.stem, fill=(0, 0, 0))
        im.save(out / f"{csv.stem}.png")
    print("previews ->", out)


if __name__ == "__main__":
    main()
