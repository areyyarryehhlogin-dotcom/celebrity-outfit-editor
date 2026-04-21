"""
Luxury Fashion Breakdown Telegram Bot
Platform: Hugging Face Spaces (Docker, CPU)
"""

import os
import threading
import logging
import telebot

from session_manager import SessionManager
from pipeline import ProcessingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_ID  = int(os.environ["OWNER_ID"])

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
sm  = SessionManager()


# ─── Guard decorator ────────────────────────────────────────────────────────

def owner_only(fn):
    def wrapper(message, *args, **kwargs):
        if message.from_user.id != OWNER_ID:
            return
        return fn(message, *args, **kwargs)
    return wrapper


# ─── /start ─────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
@owner_only
def cmd_start(message):
    bot.send_message(
        message.chat.id,
        "👗 <b>Luxury Fashion Breakdown Bot</b>\n\n"
        "<b>Step 1:</b> Send a video (≤20 MB) with caption = timestamp in seconds\n"
        "  <i>Example caption:</i> <code>5.5</code>\n\n"
        "<b>Step 2:</b> Send product entries (up to 8):\n"
        "  Format: <code>Product Name, GridCoord, Price</code>\n"
        "  <i>Example:</i> <code>Vintage White Coat, H8, $38</code>\n\n"
        "<b>Step 3:</b> After each product entry, send a photo of that item.\n\n"
        "<b>Step 4:</b> Send /done to start rendering.\n\n"
        "Send /reset to start over at any time."
    )


# ─── /reset ─────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["reset"])
@owner_only
def cmd_reset(message):
    sm.clear_session(OWNER_ID)
    bot.send_message(message.chat.id, "🔄 Session cleared. Send a new video to begin.")


# ─── /done ──────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["done"])
@owner_only
def cmd_done(message):
    session = sm.load_session(OWNER_ID)
    if not session:
        bot.send_message(message.chat.id, "⚠️ No active session. Send a video to start.")
        return

    products = session.get("products", [])
    if not products:
        bot.send_message(message.chat.id, "⚠️ Add at least one product before rendering.")
        return

    # Check that the last product has a photo
    last = products[-1]
    if not last.get("photo_path"):
        bot.send_message(message.chat.id, "⚠️ Please send the photo for your last product first.")
        return

    # Check for rendering already in progress
    if session.get("status") == "rendering":
        bot.send_message(message.chat.id, "⏳ Already rendering. Please wait.")
        return

    sm.update_status(OWNER_ID, "rendering")
    bot.send_message(
        message.chat.id,
        f"⏳ <b>Rendering your video…</b>\n\n"
        f"Products: {len(products)}\n"
        f"This will take 15–40 minutes on CPU.\n"
        f"I'll send the file when it's ready."
    )

    # Run in background thread to bypass HF 60s timeout
    t = threading.Thread(
        target=_render_and_send,
        args=(message.chat.id, session),
        daemon=True
    )
    t.start()


def _render_and_send(chat_id, session):
    try:
        pipeline = ProcessingPipeline(session)
        output_path = pipeline.run()

        file_size = os.path.getsize(output_path)
        log.info(f"Output file size: {file_size / 1024 / 1024:.1f} MB")

        if file_size <= 45 * 1024 * 1024:
            with open(output_path, "rb") as f:
                bot.send_video(
                    chat_id,
                    f,
                    caption="✨ <b>The Edit. Yours to own.</b>",
                    supports_streaming=True
                )
        else:
            # Upload to Catbox
            import requests
            with open(output_path, "rb") as f:
                resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": f},
                    timeout=300
                )
            url = resp.text.strip()
            bot.send_message(
                chat_id,
                f"✨ <b>The Edit. Yours to own.</b>\n\n"
                f"📥 <a href='{url}'>Download your video</a>\n"
                f"<i>(File was {file_size/1024/1024:.1f} MB — hosted on Catbox)</i>"
            )

        sm.update_status(OWNER_ID, "complete")
        log.info("Video delivered successfully.")

    except Exception as e:
        log.exception("Pipeline error")
        bot.send_message(chat_id, f"❌ Render failed: {e}\n\nSend /reset to start over.")
        sm.update_status(OWNER_ID, "error")


# ─── Video handler (Step 1) ──────────────────────────────────────────────────

@bot.message_handler(content_types=["video", "document"])
@owner_only
def handle_video(message):
    session = sm.load_session(OWNER_ID)
    if session and session.get("status") == "rendering":
        bot.send_message(message.chat.id, "⏳ Rendering in progress. Please wait.")
        return

    # Parse timestamp from caption
    caption = (message.caption or "").strip()
    try:
        timestamp = float(caption)
    except ValueError:
        bot.send_message(
            message.chat.id,
            "⚠️ Please add the timestamp (in seconds) as the video caption.\n"
            "Example: send the video with caption <code>5.5</code>"
        )
        return

    # Download video
    if message.content_type == "video":
        file_info = bot.get_file(message.video.file_id)
        file_size = message.video.file_size
    else:
        file_info = bot.get_file(message.document.file_id)
        file_size = message.document.file_size

    if file_size and file_size > 20 * 1024 * 1024:
        bot.send_message(message.chat.id, "⚠️ Video must be ≤ 20 MB.")
        return

    bot.send_message(message.chat.id, "⬇️ Downloading video…")

    downloaded = bot.download_file(file_info.file_path)
    video_local = f"/tmp/input_{OWNER_ID}.mp4"
    with open(video_local, "wb") as f:
        f.write(downloaded)

    # Upload to HF dataset
    hf_video_path = sm.upload_file(video_local, f"videos/input_{OWNER_ID}.mp4")

    # Init session
    sm.init_session(OWNER_ID, hf_video_path, timestamp)

    # Generate grid frame and send back
    bot.send_message(message.chat.id, "🎞️ Extracting frame and generating coordinate grid…")
    try:
        from grid_frame import generate_grid_frame
        grid_png = generate_grid_frame(video_local, timestamp)
        with open(grid_png, "rb") as f:
            bot.send_photo(
                message.chat.id,
                f,
                caption=(
                    f"📐 Frame at <b>{timestamp}s</b> with blueprint coordinate grid.\n\n"
                    f"Now send products in this format:\n"
                    f"<code>Product Name, GridCoord, Price</code>\n"
                    f"<i>Example: Vintage White Coat, H8, $38</i>"
                )
            )
    except Exception as e:
        log.exception("Grid frame error")
        bot.send_message(message.chat.id, f"⚠️ Could not generate grid frame: {e}")


# ─── Text handler (Steps 2 & product entry) ─────────────────────────────────

@bot.message_handler(content_types=["text"])
@owner_only
def handle_text(message):
    text = message.text.strip()
    if text.startswith("/"):
        return  # handled by command handlers

    session = sm.load_session(OWNER_ID)
    if not session:
        bot.send_message(message.chat.id, "⚠️ No active session. Send a video first.")
        return

    status = session.get("status")
    if status == "rendering":
        bot.send_message(message.chat.id, "⏳ Rendering in progress.")
        return

    # Expect: "Product Name, GridCoord, Price"
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 3:
        bot.send_message(
            message.chat.id,
            "⚠️ Format: <code>Product Name, GridCoord, Price</code>\n"
            "Example: <code>Vintage White Coat, H8, $38</code>"
        )
        return

    products = session.get("products", [])

    if len(products) >= 8:
        bot.send_message(message.chat.id, "⚠️ Maximum 8 products reached. Send /done to render.")
        return

    # Check last product doesn't need a photo still
    if products and not products[-1].get("photo_path"):
        bot.send_message(
            message.chat.id,
            "⚠️ Please send the photo for the previous product first."
        )
        return

    name  = parts[0]
    coord = parts[1]
    price = ",".join(parts[2:])  # price might contain comma e.g. "$1,200"

    product_entry = {
        "name": name,
        "coordinate": coord,
        "price": price,
        "photo_path": None,
        "processed_path": None
    }
    products.append(product_entry)
    session["products"] = products
    session["status"] = "awaiting_photo"
    sm.save_session(OWNER_ID, session)

    n = len(products)
    bot.send_message(
        message.chat.id,
        f"✓ <b>Product {n} saved:</b>\n"
        f"  Name: {name}\n"
        f"  Coordinate: {coord}\n"
        f"  Price: {price}\n\n"
        f"Now send the product photo."
    )


# ─── Photo handler (Step 3) ─────────────────────────────────────────────────

@bot.message_handler(content_types=["photo"])
@owner_only
def handle_photo(message):
    session = sm.load_session(OWNER_ID)
    if not session:
        bot.send_message(message.chat.id, "⚠️ No active session. Send a video first.")
        return

    products = session.get("products", [])
    if not products:
        bot.send_message(message.chat.id, "⚠️ Add a product entry first.")
        return

    last = products[-1]
    if last.get("photo_path"):
        bot.send_message(
            message.chat.id,
            "⚠️ This product already has a photo. Add the next product or send /done."
        )
        return

    # Download highest-res version
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded = bot.download_file(file_info.file_path)

    idx = len(products) - 1
    local_path = f"/tmp/product_{OWNER_ID}_{idx}.jpg"
    with open(local_path, "wb") as f:
        f.write(downloaded)

    hf_path = sm.upload_file(local_path, f"photos/product_{OWNER_ID}_{idx}.jpg")
    products[idx]["photo_path"] = hf_path
    session["products"] = products
    session["status"] = "awaiting_products"
    sm.save_session(OWNER_ID, session)

    n = idx + 1
    bot.send_message(
        message.chat.id,
        f"✓ <b>Photo {n} saved.</b>\n\n"
        f"Send another product entry or /done to render.\n"
        f"<i>({n}/8 products added)</i>"
    )


# ─── Health-check HTTP server (HF Spaces requires port 7860) ───────────────

def _run_health_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Luxury Fashion Bot - Running")
        def log_message(self, *args):
            pass
    HTTPServer(("0.0.0.0", 7860), _Handler).serve_forever()


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Health server keeps HF Space marked as Running
    threading.Thread(target=_run_health_server, daemon=True).start()
    log.info("Health server running on :7860")

    log.info("Bot starting with infinity_polling…")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
