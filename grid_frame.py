"""
Grid Frame Generator
Extracts a frame at a given timestamp and overlays a blueprint-style
coordinate grid. Returns path to a PNG file.
"""

import logging
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

GRID_COLS = 26 * 2  # A-Z then AA-AZ = 52 cols (we cap display at 50)
GRID_ROWS = 50
GRID_COLOR_DARK  = (142, 180, 212, 89)   # #8EB4D4 @ 35%
GRID_COLOR_LIGHT = (255, 255, 255, 64)   # white @ 25%

FONT_SIZE_LABEL = 18


def _col_label(idx: int) -> str:
    """0 → A, 25 → Z, 26 → AA, 27 → AB, …"""
    if idx < 26:
        return string.ascii_uppercase[idx]
    idx -= 26
    return "A" + string.ascii_uppercase[idx]


def extract_frame(video_path: str, timestamp: float) -> Image.Image:
    """Extract a single frame using OpenCV (fast, no MoviePy overhead)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp}s")
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    except Exception as e:
        raise RuntimeError(f"cv2 failed to read frame: {e}")


def _is_dark_frame(img: Image.Image) -> bool:
    """Determine if the average luminance of the frame is dark."""
    gray = np.array(img.convert("L"))
    return float(gray.mean()) < 128


def overlay_grid(img: Image.Image) -> Image.Image:
    """Overlay a 50×50 blueprint-style coordinate grid on the image."""
    w, h = img.size

    # Work on RGBA for transparency
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    dark = _is_dark_frame(img)
    line_color = GRID_COLOR_DARK if dark else GRID_COLOR_LIGHT

    cols = 50
    rows = 50
    cell_w = w / cols
    cell_h = h / rows

    # Draw vertical lines
    for c in range(cols + 1):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)

    # Draw horizontal lines
    for r in range(rows + 1):
        y = int(r * cell_h)
        draw.line([(0, y), (w, y)], fill=line_color, width=1)

    # Composite overlay
    result = Image.alpha_composite(base, overlay)

    # Add labels on outer edges only
    label_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_overlay)

    try:
        font = ImageFont.truetype("/fonts/Inter/Inter-Regular.ttf", FONT_SIZE_LABEL)
    except Exception:
        font = ImageFont.load_default()

    label_color = (255, 255, 255, 180)

    # Top edge: column labels (A, B, C…)
    for c in range(cols):
        label = _col_label(c)
        x = int((c + 0.5) * cell_w)
        y = 4
        bbox = label_draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        label_draw.text((x - tw // 2, y), label, font=font, fill=label_color)

    # Left edge: row labels (1…50)
    for r in range(rows):
        label = str(r + 1)
        x = 4
        y = int((r + 0.5) * cell_h)
        bbox = label_draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        label_draw.text((x, y - th // 2), label, font=font, fill=label_color)

    result = Image.alpha_composite(result, label_overlay)
    return result.convert("RGB")


def generate_grid_frame(video_path: str, timestamp: float) -> str:
    """
    Extract frame at timestamp, overlay coordinate grid, save as PNG.
    Returns local path to the PNG.
    """
    log.info(f"Generating grid frame at t={timestamp}s from {video_path}")
    frame = extract_frame(video_path, timestamp)
    grid_img = overlay_grid(frame)
    out_path = f"/tmp/grid_frame_{int(timestamp*10)}.png"
    grid_img.save(out_path, "PNG")
    log.info(f"Grid frame saved: {out_path}")
    return out_path


def coord_to_pixel(coord: str, frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Convert a grid coordinate like 'H8' to pixel (x, y) on the frame.
    Columns: A-Z then AA-AZ (50 total), Rows: 1-50.
    """
    coord = coord.strip().upper()

    # Parse column letters
    if len(coord) >= 2 and coord[1].isalpha():
        col_str = coord[:2]
        row_str = coord[2:]
    else:
        col_str = coord[0]
        row_str = coord[1:]

    # Column index
    if len(col_str) == 1:
        col_idx = ord(col_str) - ord('A')
    else:
        col_idx = 26 + (ord(col_str[1]) - ord('A'))

    col_idx = max(0, min(col_idx, 49))

    try:
        row_idx = int(row_str) - 1
        row_idx = max(0, min(row_idx, 49))
    except ValueError:
        row_idx = 0

    x = int((col_idx + 0.5) * frame_w / 50)
    y = int((row_idx + 0.5) * frame_h / 50)
    return x, y
