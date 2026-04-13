"""
Generate a label-audit report for duplicate groups.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
import pillow_avif

from check_duplicates import find_duplicates, get_perceptual_hash, hamming_distance


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}
THUMB_SIZE = (220, 160)
PADDING = 16
TEXT_HEIGHT = 64
BG_COLOR = "white"
BORDER_COLOR = "#d0d0d0"
TEXT_COLOR = "#111111"


def load_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def summarize_group(group: List[Path]) -> Dict:
    members = []
    hashes = {}

    for path in group:
        with Image.open(path) as img:
            hashes[str(path)] = get_perceptual_hash(img)
            members.append({
                "path": str(path),
                "class_name": path.parent.name,
                "filename": path.name,
            })

    for member in members:
        distances = []
        for other in members:
            if other["path"] == member["path"]:
                continue
            distances.append(
                hamming_distance(hashes[member["path"]], hashes[other["path"]])
            )
        member["min_distance_to_group"] = min(distances) if distances else 0
        member["max_distance_to_group"] = max(distances) if distances else 0

    class_counts = Counter(member["class_name"] for member in members)
    cross_class = len(class_counts) > 1

    return {
        "group_size": len(members),
        "classes": sorted(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "cross_class": cross_class,
        "review_priority": "high" if cross_class else "medium",
        "members": members,
    }


def create_contact_sheet(group: Dict, output_path: Path):
    font_title = load_font(18)
    font_body = load_font(14)

    cols = min(3, len(group["members"]))
    rows = math.ceil(len(group["members"]) / cols)
    cell_w = THUMB_SIZE[0] + PADDING * 2
    cell_h = THUMB_SIZE[1] + TEXT_HEIGHT + PADDING * 2
    width = cols * cell_w
    height = rows * cell_h + 50

    sheet = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(sheet)
    title = f"Duplicate Group | classes={', '.join(group['classes'])}"
    draw.text((PADDING, 14), title, fill=TEXT_COLOR, font=font_title)

    for idx, member in enumerate(group["members"]):
        row = idx // cols
        col = idx % cols
        x = col * cell_w + PADDING
        y = row * cell_h + 50

        with Image.open(member["path"]) as img:
            img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE)
            thumb = Image.new("RGB", THUMB_SIZE, "white")
            paste_x = (THUMB_SIZE[0] - img.width) // 2
            paste_y = (THUMB_SIZE[1] - img.height) // 2
            thumb.paste(img, (paste_x, paste_y))

        sheet.paste(thumb, (x, y))
        draw.rectangle(
            (x, y, x + THUMB_SIZE[0], y + THUMB_SIZE[1]),
            outline=BORDER_COLOR,
            width=1,
        )

        text_y = y + THUMB_SIZE[1] + 8
        lines = [
            f"class: {member['class_name']}",
            f"file: {member['filename']}",
            f"min dHash distance: {member['min_distance_to_group']}",
        ]
        for line in lines:
            draw.text((x, text_y), line, fill=TEXT_COLOR, font=font_body)
            text_y += 16

    sheet.save(output_path)


def write_markdown_report(groups: List[Dict], output_path: Path):
    lines = [
        "# Duplicate Label Audit",
        "",
        f"Total duplicate groups: {len(groups)}",
        f"Cross-class groups: {sum(1 for group in groups if group['cross_class'])}",
        "",
    ]

    for idx, group in enumerate(groups, 1):
        lines.append(f"## Group {idx}")
        lines.append(f"- Priority: `{group['review_priority']}`")
        lines.append(f"- Cross-class: `{group['cross_class']}`")
        lines.append(f"- Classes: `{', '.join(group['classes'])}`")
        lines.append(f"- Class counts: `{group['class_counts']}`")
        lines.append(f"- Contact sheet: `group_{idx:02d}.png`")
        lines.append("")
        for member in group["members"]:
            lines.append(
                f"- `{member['class_name']}` | `{member['filename']}` | "
                f"`{member['path']}` | min distance `{member['min_distance_to_group']}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def write_review_manifest(groups: List[Dict], output_path: Path):
    """Write an editable CSV manifest for audit decisions."""
    fieldnames = [
        "group_id",
        "priority",
        "path",
        "current_class",
        "filename",
        "action",
        "target_class",
        "notes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, group in enumerate(groups, 1):
            for member in group["members"]:
                writer.writerow({
                    "group_id": idx,
                    "priority": group["review_priority"],
                    "path": member["path"],
                    "current_class": member["class_name"],
                    "filename": member["filename"],
                    "action": "pending",
                    "target_class": "",
                    "notes": "",
                })


def generate_audit(data_dir: str, output_dir: str, similarity_threshold: int = 10):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_groups = find_duplicates(str(data_path), similarity_threshold=similarity_threshold)
    groups = [summarize_group(group) for group in raw_groups]
    groups.sort(key=lambda group: (not group["cross_class"], -group["group_size"], group["classes"]))

    for idx, group in enumerate(groups, 1):
        create_contact_sheet(group, output_path / f"group_{idx:02d}.png")

    with open(output_path / "duplicate_audit.json", "w") as f:
        json.dump(groups, f, indent=2)

    write_markdown_report(groups, output_path / "duplicate_audit.md")
    write_review_manifest(groups, output_path / "review_manifest.csv")

    print(f"Wrote audit to {output_path}")
    print(f"Duplicate groups: {len(groups)}")
    print(f"Cross-class groups: {sum(1 for group in groups if group['cross_class'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate duplicate label audit report")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="audit/duplicate_audit")
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()

    generate_audit(args.data_dir, args.output_dir, args.threshold)
