"""
Processing Pipeline — orchestrates the full render:
1. Background removal (rembg CPU)
2. Product upscaling (PIL LANCZOS)
3. Frame extraction & upscale to 2K
4. Product placement algorithm
5. Line draw animation (frame-by-frame)
6. Video assembly (MoviePy)
7. Final encode (ffmpeg subprocess)
"""

import os
import gc
import logging
import subprocess
import shutil
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from session_manager import SessionManager
from grid_frame import extract_frame, coord_to_pixel

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_W, TARGET_H = 1440, 2560   # 2K 9:16
THUMB_BOX = 320                    # max bounding box for product thumbnails
FPS = 30

# Typography sizes at 2K
FONT_PRODUCT_SIZE = 30
FONT_PRICE_SIZE   = 24
FONT_CTA_SIZE     = 64

# Colors
WHITE  = (255, 255, 255, 255)
BLACK  = (0, 0, 0, 255)
WHITE3 = (255, 255, 255)
BLACK3 = (0, 0, 0)

# Letter spacing (approx in px for PIL)
LETTER_SPACING_PRODUCT = 4
LETTER_SPACING_PRICE   = 2

SHUTTER_WAV = "/app/assets/shutter_click.wav"

# ─── Fonts ───────────────────────────────────────────────────────────────────

def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        log.warning(f"Font not found at {path}, using default")
        return ImageFont.load_default()

def get_fonts():
    playfair_path = "/fonts/Playfair_Display/PlayfairDisplay-Regular.ttf"
    inter_path    = "/fonts/Inter/Inter-Regular.ttf"
    return {
        "product": _load_font(playfair_path, FONT_PRODUCT_SIZE),
        "price":   _load_font(inter_path, FONT_PRICE_SIZE),
        "cta":     _load_font(playfair_path, FONT_CTA_SIZE),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def draw_spaced_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                     font: ImageFont.FreeTypeFont, fill, spacing: int = 0):
    """Draw text with extra letter spacing."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + spacing

def text_width_spaced(draw: ImageDraw.ImageDraw, text: str,
                      font: ImageFont.FreeTypeFont, spacing: int = 0) -> int:
    total = 0
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        total += (bbox[2] - bbox[0]) + spacing
    return total


# ─── Step 1: Background removal ──────────────────────────────────────────────

def remove_background(image_path: str, out_path: str):
    """Run rembg in CPU mode. Deletes session after use."""
    from rembg import remove, new_session
    log.info(f"rembg: removing background from {image_path}")
    session = new_session("u2net")
    with open(image_path, "rb") as f:
        input_data = f.read()
    output_data = remove(input_data, session=session)
    del session
    gc.collect()
    with open(out_path, "wb") as f:
        f.write(output_data)
    log.info(f"rembg: done → {out_path}")


# ─── Step 2: Product upscaling ───────────────────────────────────────────────

def upscale_product(image_path: str) -> Image.Image:
    """Load PNG (transparent), fit within THUMB_BOX preserving aspect."""
    img = Image.open(image_path).convert("RGBA")
    img.thumbnail((THUMB_BOX, THUMB_BOX), Image.LANCZOS)
    return img


# ─── Step 3: Frame extraction & 2K upscale ───────────────────────────────────

def get_frozen_frame_2k(video_path: str, timestamp: float) -> Image.Image:
    """Extract frame and upscale to 1440×2560 using LANCZOS."""
    frame = extract_frame(video_path, timestamp)
    # Scale to fit 9:16 canvas
    frame_w, frame_h = frame.size
    src_ratio  = frame_w / frame_h
    tgt_ratio  = TARGET_W / TARGET_H

    if src_ratio > tgt_ratio:
        # wider than target — fit by width
        new_w = TARGET_W
        new_h = int(TARGET_W / src_ratio)
    else:
        new_h = TARGET_H
        new_w = int(TARGET_H * src_ratio)

    frame = frame.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto black 2K canvas
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), BLACK3)
    paste_x = (TARGET_W - new_w) // 2
    paste_y = (TARGET_H - new_h) // 2
    canvas.paste(frame, (paste_x, paste_y))
    return canvas


# ─── Step 4: Placement algorithm ─────────────────────────────────────────────

def find_empty_regions(frame: Image.Image, n_products: int) -> list[tuple[int,int]]:
    """
    Return n_products (x, y) top-left positions for thumbnails.
    Prefers corners → edges → mid areas with low pixel variance.
    """
    gray = np.array(frame.convert("L"), dtype=float)
    H, W = gray.shape
    step = THUMB_BOX + 40
    margin = 60

    # Candidate positions (center of thumb): corners first, then edges, then interior
    candidates = []

    # Corners
    corners = [
        (margin, margin),
        (W - THUMB_BOX - margin, margin),
        (margin, H - THUMB_BOX - margin),
        (W - THUMB_BOX - margin, H - THUMB_BOX - margin),
    ]
    candidates.extend(corners)

    # Edge midpoints
    edges = [
        (W // 2 - THUMB_BOX // 2, margin),
        (W // 2 - THUMB_BOX // 2, H - THUMB_BOX - margin),
        (margin, H // 2 - THUMB_BOX // 2),
        (W - THUMB_BOX - margin, H // 2 - THUMB_BOX // 2),
    ]
    candidates.extend(edges)

    # Interior grid
    for row in range(1, 4):
        for col in range(1, 4):
            x = int(col * W / 4 - THUMB_BOX // 2)
            y = int(row * H / 4 - THUMB_BOX // 2)
            candidates.append((x, y))

    def variance_at(x, y):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + THUMB_BOX), min(H, y + THUMB_BOX)
        region = gray[y1:y2, x1:x2]
        return float(np.var(region)) if region.size > 0 else 9999

    def overlaps(placed: list, x, y) -> bool:
        for px, py in placed:
            if abs(px - x) < THUMB_BOX + 20 and abs(py - y) < THUMB_BOX + 20:
                return True
        return False

    # Score candidates by variance (lower = emptier)
    scored = sorted(candidates, key=lambda p: variance_at(p[0], p[1]))

    placed = []
    for x, y in scored:
        if len(placed) >= n_products:
            break
        # Clamp to canvas
        x = max(margin, min(x, W - THUMB_BOX - margin))
        y = max(margin, min(y, H - THUMB_BOX - margin))
        if not overlaps(placed, x, y):
            placed.append((x, y))

    # If not enough positions found, fallback to a simple grid
    while len(placed) < n_products:
        idx = len(placed)
        x = margin + (idx % 2) * (W - 2 * margin - THUMB_BOX)
        y = margin + (idx // 2) * (step)
        placed.append((x, y))

    return placed[:n_products]


# ─── Step 5: Line draw animation frames ──────────────────────────────────────

def generate_line_frames(base_frame: Image.Image,
                         start_xy: tuple[int,int],
                         end_xy: tuple[int,int],
                         n_frames: int = 18) -> list[Image.Image]:
    """
    Generate n_frames PIL images where the line draws itself from
    start_xy (product thumbnail center) to end_xy (coord dot).
    """
    frames = []
    x1, y1 = start_xy
    x2, y2 = end_xy

    for i in range(1, n_frames + 1):
        t = i / n_frames
        cx = int(x1 + (x2 - x1) * t)
        cy = int(y1 + (y2 - y1) * t)

        img = base_frame.copy().convert("RGBA")
        draw = ImageDraw.Draw(img)
        draw.line([(x1, y1), (cx, cy)], fill=WHITE, width=1)

        # Terminal dot at coordinate end only when fully drawn
        if i == n_frames:
            r = 4
            draw.ellipse(
                [(x2 - r, y2 - r), (x2 + r, y2 + r)],
                fill=WHITE
            )

        frames.append(img.convert("RGB"))

    return frames


# ─── Fade helpers ─────────────────────────────────────────────────────────────

def fade_composite(base: Image.Image, overlay: Image.Image, alpha: float) -> Image.Image:
    """Blend overlay onto base at alpha (0.0–1.0)."""
    base_a = base.convert("RGBA")
    ov_a   = overlay.convert("RGBA")
    # Apply alpha to overlay
    r, g, b, a = ov_a.split()
    a = a.point(lambda v: int(v * alpha))
    ov_a = Image.merge("RGBA", (r, g, b, a))
    result = Image.alpha_composite(base_a, ov_a)
    return result.convert("RGB")

def solid_color_frame(color: tuple, size: tuple) -> Image.Image:
    return Image.new("RGB", size, color)


# ─── Video writing helper ─────────────────────────────────────────────────────

class VideoWriter:
    """Write frames as individual PNGs into a temp dir for ffmpeg ingestion."""

    def __init__(self, out_dir: str, fps: int = FPS):
        self.out_dir = out_dir
        self.fps = fps
        self.idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def write(self, frame: Image.Image):
        path = os.path.join(self.out_dir, f"frame_{self.idx:06d}.png")
        frame.save(path, "PNG")
        self.idx += 1
        del frame

    def write_n(self, frame: Image.Image, n: int):
        for _ in range(n):
            self.write(frame.copy())

    def duration_frames(self, seconds: float) -> int:
        return max(1, int(seconds * self.fps))


# ─── CTA screen ──────────────────────────────────────────────────────────────

def make_cta_frames(fonts: dict, writer: VideoWriter):
    """Generate CTA screen frames: fade in, hold, fade out."""
    black = solid_color_frame(BLACK3, (TARGET_W, TARGET_H))
    cta_text = "The Edit. Yours to own."
    font = fonts["cta"]

    # Build text image
    text_img = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_img)
    tw = text_width_spaced(td, cta_text, font, spacing=4)
    tx = (TARGET_W - tw) // 2
    ty = TARGET_H // 2 - FONT_CTA_SIZE // 2
    draw_spaced_text(td, (tx, ty), cta_text, font, WHITE, spacing=4)

    fade_in_frames  = writer.duration_frames(0.8)
    hold_frames     = writer.duration_frames(2.5)
    fade_out_frames = writer.duration_frames(0.5)

    # Fade in
    for i in range(fade_in_frames):
        alpha = (i + 1) / fade_in_frames
        frame = fade_composite(black, text_img, alpha)
        writer.write(frame)

    # Hold
    full_frame = fade_composite(black, text_img, 1.0)
    writer.write_n(full_frame, hold_frames)

    # Fade out
    for i in range(fade_out_frames):
        alpha = 1.0 - (i + 1) / fade_out_frames
        frame = fade_composite(black, text_img, alpha)
        writer.write(frame)


# ─── Product overlay renderer ────────────────────────────────────────────────

def render_product_overlay(frozen_2k: Image.Image,
                            product_thumb: Image.Image,
                            thumb_pos: tuple[int,int],
                            coord_px: tuple[int,int],
                            product: dict,
                            fonts: dict,
                            writer: VideoWriter):
    """
    Render one product reveal sequence onto a running base_frame.
    Returns the updated base_frame (with this product permanently composited).
    """
    tx, ty = thumb_pos
    tw, th = product_thumb.size

    # ── a. Line draws itself (18 frames = 0.6s at 30fps)
    line_start = (tx + tw // 2, ty + th // 2)
    line_frames = generate_line_frames(frozen_2k, line_start, coord_px, n_frames=18)
    for lf in line_frames:
        writer.write(lf)
    del line_frames
    gc.collect()

    # Base with line drawn (last frame of line animation)
    with_line = frozen_2k.copy().convert("RGBA")
    draw_l = ImageDraw.Draw(with_line)
    draw_l.line([line_start, coord_px], fill=WHITE, width=1)
    r = 4
    draw_l.ellipse(
        [(coord_px[0]-r, coord_px[1]-r), (coord_px[0]+r, coord_px[1]+r)],
        fill=WHITE
    )
    with_line = with_line.convert("RGB")

    # Build the full product overlay (thumb + name + price) on transparent layer
    product_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    pl_draw = ImageDraw.Draw(product_layer)

    # Paste thumbnail
    product_layer.paste(product_thumb, (tx, ty), product_thumb)

    # Product name above
    name = product["name"]
    nw = text_width_spaced(pl_draw, name, fonts["product"], LETTER_SPACING_PRODUCT)
    nx = tx + tw // 2 - nw // 2
    ny = ty - FONT_PRODUCT_SIZE - 8

    # Price below
    price = product["price"]
    pw = text_width_spaced(pl_draw, price, fonts["price"], LETTER_SPACING_PRICE)
    px = tx + tw // 2 - pw // 2
    py = ty + th + 6

    # ── b. Thumbnail fades in (21 frames = 0.7s)
    fade_thumb_n = writer.duration_frames(0.7)
    for i in range(fade_thumb_n):
        alpha = (i + 1) / fade_thumb_n
        # Only thumb in layer for this phase
        thumb_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        thumb_layer.paste(product_thumb, (tx, ty), product_thumb)
        frame = fade_composite(with_line, thumb_layer, alpha)
        writer.write(frame)

    with_thumb = with_line.copy()
    tl = Image.new("RGBA", with_thumb.size, (0,0,0,0))
    tl.paste(product_thumb, (tx, ty), product_thumb)
    with_thumb = Image.alpha_composite(with_thumb.convert("RGBA"), tl).convert("RGB")

    # ── c. Product name fades in (12 frames = 0.4s)
    fade_name_n = writer.duration_frames(0.4)
    for i in range(fade_name_n):
        alpha = (i + 1) / fade_name_n
        name_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        ndraw = ImageDraw.Draw(name_layer)
        draw_spaced_text(ndraw, (nx, ny), name, fonts["product"], WHITE, LETTER_SPACING_PRODUCT)
        frame = fade_composite(with_thumb, name_layer, alpha)
        writer.write(frame)

    with_name = with_thumb.copy()
    nl = Image.new("RGBA", with_name.size, (0,0,0,0))
    ndraw2 = ImageDraw.Draw(nl)
    draw_spaced_text(ndraw2, (nx, ny), name, fonts["product"], WHITE, LETTER_SPACING_PRODUCT)
    with_name = Image.alpha_composite(with_name.convert("RGBA"), nl).convert("RGB")

    # ── d. Price fades in (9 frames = 0.3s)
    fade_price_n = writer.duration_frames(0.3)
    for i in range(fade_price_n):
        alpha = (i + 1) / fade_price_n
        price_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(price_layer)
        draw_spaced_text(pdraw, (px, py), price, fonts["price"], WHITE, LETTER_SPACING_PRICE)
        frame = fade_composite(with_name, price_layer, alpha)
        writer.write(frame)

    # Build final frame with all elements of this product
    final = with_name.copy()
    pl = Image.new("RGBA", final.size, (0,0,0,0))
    pdraw2 = ImageDraw.Draw(pl)
    draw_spaced_text(pdraw2, (px, py), price, fonts["price"], WHITE, LETTER_SPACING_PRICE)
    final = Image.alpha_composite(final.convert("RGBA"), pl).convert("RGB")

    # ── e. 0.3s pause
    pause_n = writer.duration_frames(0.3)
    writer.write_n(final, pause_n)

    gc.collect()
    return final  # Caller accumulates this as the running base


# ─── Final ffmpeg encode ──────────────────────────────────────────────────────

def ffmpeg_encode(frames_dir: str, audio_path,
                  output_path: str, fps: int = FPS, crf: int = 18):
    """Encode frame sequence + optional audio with ffmpeg subprocess."""
    pattern = os.path.join(frames_dir, "frame_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
    ]

    if audio_path and os.path.exists(audio_path):
        cmd += ["-i", audio_path, "-shortest"]

    cmd += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2",
    ]

    if audio_path and os.path.exists(audio_path):
        cmd += ["-c:a", "aac", "-ar", "44100", "-b:a", "192k"]

    cmd.append(output_path)

    log.info(f"ffmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffmpeg stderr: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    log.info("ffmpeg encode complete")


# ─── Main pipeline class ──────────────────────────────────────────────────────

class ProcessingPipeline:
    def __init__(self, session: dict):
        self.session = session
        self.sm = SessionManager()
        self.fonts = get_fonts()
        self.frames_dir = tempfile.mkdtemp(prefix="fashion_frames_")

    def _download_video(self) -> str:
        hf_path = self.session["video_path"]
        local = "/tmp/pipeline_input.mp4"
        self.sm.download_hf_file(hf_path, local)
        return local

    def _download_photos(self) -> list[str]:
        paths = []
        for i, p in enumerate(self.session["products"]):
            local = f"/tmp/pipeline_photo_{i}.jpg"
            self.sm.download_hf_file(p["photo_path"], local)
            paths.append(local)
        return paths

    def run(self) -> str:
        log.info("Pipeline starting…")
        writer = VideoWriter(self.frames_dir, FPS)

        # ── Download assets
        video_path = self._download_video()
        photo_paths = self._download_photos()
        timestamp = float(self.session["timestamp"])
        products  = self.session["products"]

        # ── Step 1 & 2: rembg + upscale each product photo
        thumbs = []
        for i, photo_path in enumerate(photo_paths):
            nobg_path = f"/tmp/nobg_{i}.png"
            remove_background(photo_path, nobg_path)
            thumb = upscale_product(nobg_path)
            thumbs.append(thumb)
            gc.collect()

        # ── Step 3: Frozen frame at 2K
        frozen_2k = get_frozen_frame_2k(video_path, timestamp)

        # ── Step 4: Placement positions
        positions = find_empty_regions(frozen_2k, len(products))

        # ── Coordinate pixel positions
        coord_pixels = []
        for prod in products:
            px, py = coord_to_pixel(prod["coordinate"], TARGET_W, TARGET_H)
            coord_pixels.append((px, py))

        # ── Part 1: Original video (0 → timestamp)
        log.info("Writing Part 1: original video frames")
        self._write_original_segment(video_path, timestamp, writer)

        # ── Transition: 0.1s white frame
        log.info("Writing transition frames")
        white_frame = solid_color_frame(WHITE3, (TARGET_W, TARGET_H))
        writer.write_n(white_frame, writer.duration_frames(0.1))
        del white_frame

        # ── Freeze moment: 0.5s clean frozen frame
        log.info("Writing freeze moment")
        writer.write_n(frozen_2k, writer.duration_frames(0.5))

        # ── Product reveals (sequential)
        log.info("Writing product reveal frames")
        running_base = frozen_2k.copy()
        for i, (product, thumb, pos, cpx) in enumerate(
            zip(products, thumbs, positions, coord_pixels)
        ):
            log.info(f"  Product {i+1}/{len(products)}: {product['name']}")
            running_base = render_product_overlay(
                running_base, thumb, pos, cpx, product, self.fonts, writer
            )
            gc.collect()

        # ── Hold: 1.5s all products visible
        log.info("Writing hold frames")
        writer.write_n(running_base, writer.duration_frames(1.5))

        # ── Fade to black: 1.0s
        log.info("Writing fade to black")
        fade_n = writer.duration_frames(1.0)
        black = solid_color_frame(BLACK3, (TARGET_W, TARGET_H))
        for i in range(fade_n):
            alpha = (i + 1) / fade_n
            blended = Image.blend(running_base, black, alpha)
            writer.write(blended)

        # ── CTA screen
        log.info("Writing CTA screen")
        make_cta_frames(self.fonts, writer)

        # ── Build audio track
        audio_path = self._build_audio_track(timestamp)

        # ── Final encode
        output_path = "/tmp/fashion_output.mp4"

        log.info("Starting ffmpeg encode (CRF 18)…")
        ffmpeg_encode(self.frames_dir, audio_path, output_path, crf=18)

        file_size = os.path.getsize(output_path)
        log.info(f"Output size: {file_size/1024/1024:.1f} MB")

        if file_size > 45 * 1024 * 1024:
            log.info("File > 45MB, re-encoding at CRF 20…")
            output_path_20 = "/tmp/fashion_output_crf20.mp4"
            ffmpeg_encode(self.frames_dir, audio_path, output_path_20, crf=20)
            output_path = output_path_20

        # Cleanup frames dir
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        log.info("Pipeline complete.")
        return output_path

    def _write_original_segment(self, video_path: str, timestamp: float, writer: VideoWriter):
        """Extract frames from 0 to timestamp and write them at TARGET resolution."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        end_frame = int(timestamp * fps_src)

        frame_idx = 0
        every_n = max(1, int(fps_src / FPS))  # downsample to target FPS

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > end_frame:
                break
            if frame_idx % every_n == 0:
                frame_rgb = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                # Fit to 2K vertical canvas
                fw, fh = frame_rgb.size
                new_w = TARGET_W
                new_h = int(TARGET_W * fh / fw)
                frame_rgb = frame_rgb.resize((new_w, new_h), Image.LANCZOS)
                canvas = Image.new("RGB", (TARGET_W, TARGET_H), BLACK3)
                paste_y = (TARGET_H - new_h) // 2
                canvas.paste(frame_rgb, (0, max(0, paste_y)))
                writer.write(canvas)
                del canvas, frame_rgb
            frame_idx += 1

        cap.release()
        gc.collect()

    def _build_audio_track(self, timestamp: float):
        """
        Build audio: silence for Part 1 duration, then shutter click, 
        then silence for the rest. Returns path to .wav or None.
        """
        try:
            import wave

            part1_duration = timestamp

            # Load shutter WAV
            if not os.path.exists(SHUTTER_WAV):
                log.warning("Shutter WAV not found, skipping audio")
                return None

            sample_rate = 44100
            audio_out = "/tmp/fashion_audio.wav"

            def silence_frames(duration: float):
                return b'\x00\x00' * int(sample_rate * duration)

            with wave.open(SHUTTER_WAV, 'rb') as sw:
                shutter_data = sw.readframes(sw.getnframes())

            silence_before = silence_frames(part1_duration)

            with wave.open(audio_out, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(silence_before)
                # Resample shutter if needed (simple: just append as-is)
                wf.writeframes(shutter_data[:sample_rate * 2])  # max 1s of shutter
                # Silence for rest
                rest_silence = silence_frames(60.0)  # plenty of trailing silence
                wf.writeframes(rest_silence)

            return audio_out

        except Exception as e:
            log.warning(f"Audio build failed: {e}")
            return None
