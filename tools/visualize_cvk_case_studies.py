#!/usr/bin/env python
"""Create qualitative panels for improved/degraded CVK retrieval cases."""
import argparse
import csv
import os
from typing import List

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--topk", default=8, type=int)
    return parser.parse_args()


def load_rows(csv_path: str):
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["delta_ap"] = float(row["delta_ap"])
        row["rank_gain"] = int(row["rank_gain"])
        row["eq_top1_correct"] = row["eq_top1_correct"] == "True"
        row["cvk_top1_correct"] = row["cvk_top1_correct"] == "True"
    return rows


def _open_img(path: str, size):
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def _draw_labeled_image(canvas, img, x, y, label, border_color):
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img, (x, y))
    draw.rectangle((x, y, x + img.width - 1, y + img.height - 1), outline=border_color, width=4)
    draw.text((x, y - 18), label, fill=(0, 0, 0), font=ImageFont.load_default())


def build_panel(rows: List[dict], title: str, out_path: str):
    if not rows:
        return
    font = ImageFont.load_default()
    img_w, img_h = 128, 384
    pad = 18
    row_h = img_h + 70
    width = pad + 3 * (img_w + pad)
    height = 60 + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), color=(250, 248, 243))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 20), title, fill=(0, 0, 0), font=font)

    y = 60
    for idx, row in enumerate(rows):
        header = (
            f"q{row['query_idx']}  status={row['status']}  "
            f"delta_ap={row['delta_ap']:+.4f}  rank_gain={row['rank_gain']:+d}"
        )
        draw.text((pad, y), header, fill=(0, 0, 0), font=font)
        img_y = y + 20

        q_img = _open_img(row["query_path"], (img_w, img_h))
        eq_img = _open_img(row["eq_top1_gallery_path"], (img_w, img_h))
        cvk_img = _open_img(row["cvk_top1_gallery_path"], (img_w, img_h))

        _draw_labeled_image(canvas, q_img, pad, img_y, "Query", (54, 84, 140))
        _draw_labeled_image(
            canvas,
            eq_img,
            pad * 2 + img_w,
            img_y,
            f"EQ top1 ({'OK' if row['eq_top1_correct'] else 'WRONG'})",
            (28, 150, 80) if row["eq_top1_correct"] else (190, 50, 40),
        )
        _draw_labeled_image(
            canvas,
            cvk_img,
            pad * 3 + img_w * 2,
            img_y,
            f"CVK top1 ({'OK' if row['cvk_top1_correct'] else 'WRONG'})",
            (28, 150, 80) if row["cvk_top1_correct"] else (190, 50, 40),
        )
        y += row_h

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    improved = sorted(rows, key=lambda r: r["delta_ap"], reverse=True)
    degraded = sorted(rows, key=lambda r: r["delta_ap"])

    improved_rows = []
    for row in improved:
        if row["status"] in {"top1_fixed", "both_top1_wrong", "both_top1_correct"}:
            improved_rows.append(row)
        if len(improved_rows) >= args.topk:
            break

    degraded_rows = []
    for row in degraded:
        if row["status"] in {"top1_degraded", "both_top1_wrong", "both_top1_correct"}:
            degraded_rows.append(row)
        if len(degraded_rows) >= args.topk:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    build_panel(
        improved_rows,
        "CVK Top Improved Cases",
        os.path.join(args.output_dir, "top_improved.png"),
    )
    build_panel(
        degraded_rows,
        "CVK Top Degraded Cases",
        os.path.join(args.output_dir, "top_degraded.png"),
    )

    print(os.path.join(args.output_dir, "top_improved.png"))
    print(os.path.join(args.output_dir, "top_degraded.png"))


if __name__ == "__main__":
    main()
