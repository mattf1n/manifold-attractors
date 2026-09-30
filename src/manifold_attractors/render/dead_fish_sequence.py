# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=12.0"]
# ///
"""Horizontal six-frame fish sequence from the supplied GIF, without annotations.

The reference fish montage is Figure 4 (printed page 391) in
'Passive propulsion in vortex wakes', not its apparatus Figure 8.
GIF frames are selected to show the corresponding bend/upstream-motion sequence;
the GIF playback times are not asserted to match the paper's experiment times.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BACKGROUND = (240, 240, 240)  # Match matplotlibrc and the adjacent TikZ plot.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--data", type=Path, default=ROOT / "assets/A-Dead-Fish-Swims.gif")
    parser.add_argument("--output", "--figure", type=Path, default=ROOT / "fig/shared/dead_fish_sequence.pdf")
    parser.add_argument("--frames", nargs="+", type=int, default=[54, 56, 58, 60, 62, 64])
    parser.add_argument("--crop", nargs=4, type=int, default=[45, 65, 345, 175],
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--width", type=float, default=5.5, help="Output width in inches")
    args = parser.parse_args()
    with Image.open(args.source) as gif:
        left, top, right, bottom = args.crop
        if not (0 <= left < right <= gif.width and 0 <= top < bottom <= gif.height):
            parser.error("Crop must be inside the source image")
        if args.width <= 0 or args.gap < 0:
            parser.error("Width must be positive and gap nonnegative")
        frames = []
        for index in args.frames:
            if not 0 <= index < gif.n_frames:
                parser.error(f"Frame {index} outside GIF frame range")
            gif.seek(index)
            # The fish advances leftward in the source; clockwise rotation
            # makes its swimming direction point toward the top of each panel.
            frame = gif.convert("RGB").crop(args.crop).transpose(Image.Transpose.ROTATE_270)
            # Map white to the paper's light gray, preserving black and smooth
            # tonal transitions rather than thresholding the silhouettes.
            frames.append(frame.point([round(value * 240 / 255) for value in range(256)] * 3))
    w, h = frames[0].size
    canvas_width = len(frames) * w + (len(frames) - 1) * args.gap
    # Match the adjacent 2:1 plot, adding gray space only below the frames.
    canvas_height = max(h, round(canvas_width / 2))
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    for i, frame in enumerate(frames):
        canvas.paste(frame, (i * (w + args.gap), 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output.with_suffix(".png"))
    canvas.save(args.output, resolution=canvas.width / args.width)
    args.output.with_suffix(".render.json").write_text(json.dumps({
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "frames_zero_based": args.frames,
        "crop_ltrb_pixels": args.crop,
        "rotation_clockwise_degrees": 90,
        "gap_pixels": args.gap,
        "background_rgb": BACKGROUND,
        "bottom_padding_pixels": canvas_height - h,
        "tone_mapping": "Each RGB channel multiplied by 240/255; black preserved, white mapped to #f0f0f0",
        "width_inches": args.width,
        "note": "Chronological GIF frames, not synchronized to published timestamps; rotated clockwise, no overlays or image warping.",
    }, indent=2) + "\n")
    print(f"Wrote {args.output} and PNG preview")


if __name__ == "__main__":
    main()
