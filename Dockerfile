FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    wget \
    unzip \
    curl \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# ── Font directories ─────────────────────────────────────────────────────────
RUN mkdir -p /fonts/Playfair_Display /fonts/Inter

# ── Download fonts via Python (handles redirects + fallback automatically) ───
COPY download_fonts.py /tmp/download_fonts.py
RUN python /tmp/download_fonts.py

# ── App setup ────────────────────────────────────────────────────────────────
RUN mkdir -p /app/assets
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── rembg: set U2NET_HOME before caching so it lands in a botuser-readable path
ENV U2NET_HOME=/app/.u2net
RUN mkdir -p /app/.u2net && \
    python -c "from rembg import new_session; s = new_session('u2net'); del s; print('rembg model cached OK')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY . /app/

# ── Generate synthetic shutter click WAV ─────────────────────────────────────
RUN python /app/generate_shutter.py

# ── Permissions: /tmp must be writable by all users ─────────────────────────
RUN chmod 1777 /tmp && \
    mkdir -p /tmp/hf_sessions /tmp/hf_downloads && \
    chmod 777 /tmp/hf_sessions /tmp/hf_downloads

# ── Non-root user (HF Spaces best practice) ──────────────────────────────────
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app /fonts

USER botuser

# ── HF Spaces requires something on port 7860 ────────────────────────────────
EXPOSE 7860

CMD ["python", "bot.py"]
