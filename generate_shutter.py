"""
generate_shutter.py
Generates a synthetic DSLR-style mechanical shutter click WAV.
Runs at Docker build time. Output: /app/assets/shutter_click.wav
Falls back gracefully if audio libs are unavailable.
"""

import wave
import struct
import math
import os

OUTPUT_PATH = "/app/assets/shutter_click.wav"
SAMPLE_RATE = 44100


def generate_click():
    """
    Synthesize a DSLR shutter click:
    - Sharp transient attack (~3ms) 
    - Mechanical noise burst (~20ms)
    - Quick decay (~30ms)
    Total: ~53ms
    """
    duration_ms = 55
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    samples = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        t_ms = t * 1000

        # Attack: sharp click (0–3ms)
        if t_ms < 3:
            val = math.sin(2 * math.pi * 8000 * t) * (1.0 - t_ms / 3.0)

        # Mechanical burst: noise-like (3–23ms)
        elif t_ms < 23:
            progress = (t_ms - 3) / 20.0
            decay = 1.0 - progress
            # Combine multiple frequencies for mechanical texture
            val = (
                math.sin(2 * math.pi * 3200 * t) * 0.5 +
                math.sin(2 * math.pi * 1800 * t) * 0.3 +
                math.sin(2 * math.pi * 6400 * t) * 0.2
            ) * decay

        # Tail: quick decay (23–55ms)
        else:
            progress = (t_ms - 23) / 32.0
            decay = max(0, 1.0 - progress)
            val = math.sin(2 * math.pi * 1200 * t) * decay * 0.3

        # Scale to 16-bit range
        sample_int = max(-32767, min(32767, int(val * 28000)))
        samples.append(sample_int)

    return samples


def save_wav(samples: list[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        data = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(data)
    print(f"Shutter click saved: {path} ({len(samples)} samples)")


if __name__ == "__main__":
    try:
        samples = generate_click()
        save_wav(samples, OUTPUT_PATH)
    except Exception as e:
        print(f"Shutter generation failed: {e}")
        # Create a silent placeholder so the pipeline doesn't crash
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        save_wav([0] * 100, OUTPUT_PATH)
