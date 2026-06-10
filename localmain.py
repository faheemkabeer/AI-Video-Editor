"""
AI Video Editor - Windows / VS Code Version
============================================
Requirements: see requirements.txt
Run: python ai_video_editor.py
"""

import os
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
import moviepy.video.fx.all as vfx_all
import moviepy.audio.fx.all as afx

# ── Windows: tell MoviePy where ImageMagick is ──────────────────────────────
# Download ImageMagick from https://imagemagick.org/script/download.php#windows
# and update this path if your install location differs.
import moviepy.config as mp_config

IMAGEMAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
if os.path.exists(IMAGEMAGICK_PATH):
    mp_config.change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_PATH})
else:
    print(
        "⚠️  ImageMagick not found at the expected path.\n"
        "   TextClip (title card) will not work.\n"
        "   Download from https://imagemagick.org and update IMAGEMAGICK_PATH."
    )

# ── Folder paths ─────────────────────────────────────────────────────────────
TRANSITION_FOLDER   = r"D:\Ai video editor\transition"
WEDDING_SONGS_DIR   = r"D:\Ai video editor\pattt\weddind"
NON_WEDDING_SONGS_DIR = r"D:\Ai video editor\pattt\non weddind"
VIDEO_FOLDER        = r"D:\Ai video editor\videos"
OUTPUT_NAME         = r"D:\Ai video editor\final.mp4"

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_EVERY_SEC    = 1.5
PREVIEW_DURATION    = 10      # seconds of audio preview
TRANSITION_FRAMES   = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU CHECK
# ═══════════════════════════════════════════════════════════════════════════════

use_cuda_cv = cv2.cuda.getCudaEnabledDeviceCount() > 0
print("CUDA (OpenCV):", "enabled ✅" if use_cuda_cv else "not available — using CPU")


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def resize_frame(frame, width=None, height=None, fx=None, fy=None):
    if not use_cuda_cv:
        if width and height:
            return cv2.resize(frame, (width, height))
        return cv2.resize(frame, None, fx=fx, fy=fy)

    gpu = cv2.cuda_GpuMat()
    gpu.upload(frame)
    if width and height:
        resized = cv2.cuda.resize(gpu, (width, height))
    else:
        resized = cv2.cuda.resize(gpu, None, fx=fx, fy=fy)
    return resized.download()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSITION METHOD 1 — Pure OpenCV (no transition clip needed)
# ═══════════════════════════════════════════════════════════════════════════════

def transition_without_video(video1, video2, output):
    """Joins two clips with a random zoom/dissolve effect using OpenCV."""

    def zoom_crossfade(buf1, buf2, w, h):
        frames = []
        n = len(buf1)
        for i in range(n):
            scale = 1 + (i / n) * 0.5
            zoomed = resize_frame(buf1[i], fx=scale, fy=scale)
            zh, zw = zoomed.shape[:2]
            sx, sy = (zw - w) // 2, (zh - h) // 2
            crop = zoomed[sy:sy+h, sx:sx+w]
            alpha = i / n
            frames.append(cv2.addWeighted(crop, 1-alpha, buf2[i], alpha, 0))
        return frames

    def zoom_in(buf, w, h):
        frames = []
        n = len(buf)
        for i in range(n):
            scale = 1 + (i / n) * 2
            zoomed = resize_frame(buf[i], fx=scale, fy=scale)
            zh, zw = zoomed.shape[:2]
            sx, sy = (zw - w) // 2, (zh - h) // 2
            frames.append(zoomed[sy:sy+h, sx:sx+w])
        return frames

    def dissolve(buf1, buf2):
        frames = []
        n = len(buf1)
        for i in range(n):
            alpha = i / n
            frames.append(cv2.addWeighted(buf1[i], 1-alpha, buf2[i], alpha, 0))
        return frames

    def pull_in(buf, w, h):
        frames = []
        n = len(buf)
        for i in range(n):
            scale = 1 + (i / n) ** 2 * 3
            zoomed = resize_frame(buf[i], fx=scale, fy=scale)
            zh, zw = zoomed.shape[:2]
            sx, sy = (zw - w) // 2, (zh - h) // 2
            frames.append(zoomed[sy:sy+h, sx:sx+w])
        return frames

    def pull_out(buf, w, h):
        frames = []
        n = len(buf)
        for i in range(n):
            scale = max(2 - i / n, 1.01)
            zoomed = resize_frame(buf[i], fx=scale, fy=scale)
            zh, zw = zoomed.shape[:2]
            sx, sy = (zw - w) // 2, (zh - h) // 2
            frames.append(zoomed[sy:sy+h, sx:sx+w])
        return frames

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    fps    = int(cap1.get(cv2.CAP_PROP_FPS))
    width  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Write all but last TRANSITION_FRAMES frames from clip 1
    for _ in range(total1 - TRANSITION_FRAMES):
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)

    # Buffer the overlap frames
    buf1, buf2 = [], []
    for _ in range(TRANSITION_FRAMES):
        ret, frame = cap1.read()
        if ret:
            buf1.append(frame)
    for _ in range(TRANSITION_FRAMES):
        ret, frame = cap2.read()
        if ret:
            buf2.append(resize_frame(frame, width, height))

    chosen = random.choice(["zoom_crossfade", "zoom_in", "dissolve", "pull_in", "pull_out"])
    print(f"  Selected transition: {chosen}")

    if   chosen == "zoom_crossfade": frames = zoom_crossfade(buf1, buf2, width, height)
    elif chosen == "zoom_in":        frames = zoom_in(buf1, width, height)
    elif chosen == "dissolve":       frames = dissolve(buf1, buf2)
    elif chosen == "pull_in":        frames = pull_in(buf1, width, height)
    else:                            frames = pull_out(buf2, width, height)

    for f in frames:
        out.write(f)

    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        out.write(resize_frame(frame, width, height))

    cap1.release(); cap2.release(); out.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSITION METHOD 2 — Overlay a transition clip (MoviePy)
# ═══════════════════════════════════════════════════════════════════════════════

def transition_with_video(video1, video2, output):
    """Overlays a random transition clip between two videos."""

    if not os.path.isdir(TRANSITION_FOLDER):
        print(f"  ⚠️  Transition folder not found: {TRANSITION_FOLDER}  — falling back to simple concat")
        simple_concat(video1, video2, output)
        return

    transition_files = [
        os.path.join(TRANSITION_FOLDER, f)
        for f in os.listdir(TRANSITION_FOLDER)
        if f.lower().endswith((".mp4", ".mov", ".mkv"))
    ]
    if not transition_files:
        print("  ⚠️  No transition clips found — falling back to simple concat")
        simple_concat(video1, video2, output)
        return

    videoA = VideoFileClip(video1).without_audio()
    videoB = VideoFileClip(video2).without_audio()

    transition_path = random.choice(transition_files)
    transition = VideoFileClip(transition_path).resize(videoA.size)
    videoB = videoB.resize(videoA.size)

    TDUR = min(1.5, transition.duration)
    transition = transition.subclip(0, TDUR)

    start = videoA.duration - TDUR
    transition = transition.set_start(start)
    videoB = videoB.set_start(start).crossfadein(TDUR)

    final = CompositeVideoClip([videoA, transition, videoB], size=videoA.size)
    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=videoA.fps)
    videoA.close(); videoB.close(); transition.close(); final.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSITION METHOD 3 — Plain concatenation
# ═══════════════════════════════════════════════════════════════════════════════

def simple_concat(video1, video2, output):
    clip1 = clip2 = final_clip = None
    try:
        clip1 = VideoFileClip(video1)
        clip2 = VideoFileClip(video2)
        if clip1.size != clip2.size:
            clip2 = clip2.resize(clip1.size)
        if clip1.fps != clip2.fps:
            clip2 = clip2.set_fps(clip1.fps)
        final_clip = concatenate_videoclips([clip1, clip2], method="compose")
        final_clip.write_videofile(
            output, codec="libx264", audio_codec="aac",
            temp_audiofile="temp-audio.m4a", remove_temp=True,
            threads=4, preset="medium"
        )
        print("  ✅ Concatenation done")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    finally:
        for c in [clip1, clip2, final_clip]:
            if c:
                try: c.close()
                except: pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP ANALYSIS WITH CLIP MODEL
# ═══════════════════════════════════════════════════════════════════════════════

THEMES = [
    "wedding ceremony","bride","groom","dance","kiss",
    "group of people","party","ring exchange","cake cutting",
    "speeches","couple portrait","crowd","outdoor","indoor",
    "dinner","reception","romantic","family","friends","music performance",
]

def analyze_theme(clip: VideoFileClip) -> tuple[str, float]:
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}…")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()

    timestamps = np.arange(0, clip.duration, SAMPLE_EVERY_SEC)
    scores = np.zeros(len(THEMES))

    for t in tqdm(timestamps, desc="Analyzing frames"):
        frame = clip.get_frame(t)
        image = Image.fromarray(frame.astype("uint8"), "RGB")
        inputs = processor(text=THEMES, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            scores += logits

    scores /= scores.sum()
    idx = int(np.argmax(scores))
    return THEMES[idx], float(scores[idx])


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO PREVIEW (Windows — uses pygame)
# ═══════════════════════════════════════════════════════════════════════════════

def preview_audio_windows(song_path: str, duration: int = PREVIEW_DURATION):
    """Play a short preview of an audio file using pygame."""
    try:
        import pygame
    except ImportError:
        print("  pygame not installed — skipping audio preview (pip install pygame)")
        return

    # Export a short temp preview
    preview_path = os.path.join(os.path.dirname(OUTPUT_NAME), "_preview_tmp.mp3")
    clip = AudioFileClip(song_path).subclip(0, min(duration, AudioFileClip(song_path).duration))
    clip.write_audiofile(preview_path, codec="libmp3lame", fps=44100, logger=None)
    clip.close()

    pygame.mixer.init()
    pygame.mixer.music.load(preview_path)
    pygame.mixer.music.play()
    print(f"  ▶ Playing {duration}s preview… (press Enter to stop and continue)")
    input()
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    try:
        os.remove(preview_path)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SONG PICKER
# ═══════════════════════════════════════════════════════════════════════════════

WEDDING_KEYWORDS = ["wedding","bride","groom","couple","reception","ceremony"]

def pick_multiple_audios(folder_path: str, n: int = 3) -> list[str]:
    exts = (".mp3", ".wav", ".m4a", ".mpeg", ".mpg")
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(exts)
    ]
    random.shuffle(files)
    return files[:n]


def choose_song(top_theme: str) -> str:
    if any(k in top_theme.lower() for k in WEDDING_KEYWORDS):
        candidates = pick_multiple_audios(WEDDING_SONGS_DIR, 3)
        label = "Wedding"
    else:
        candidates = pick_multiple_audios(NON_WEDDING_SONGS_DIR, 3)
        label = "Non-wedding"

    if not candidates:
        raise ValueError(f"❌ No audio files found in {label} folder!")

    print(f"\n🎵 Suggested {label} songs:")
    for i, song in enumerate(candidates, 1):
        print(f"  {i}. {os.path.basename(song)}")

    for i, song in enumerate(candidates, 1):
        print(f"\n▶ Previewing song {i}: {os.path.basename(song)}")
        preview_audio_windows(song)

    while True:
        try:
            choice = int(input("\nEnter the number of the song to use: ")) - 1
            if 0 <= choice < len(candidates):
                break
            print("  Invalid number, try again.")
        except ValueError:
            print("  Please enter a number.")

    selected = candidates[choice]
    print(f"✅ Selected: {os.path.basename(selected)}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Collect videos ────────────────────────────────────────────────────
    folder = input(f"Enter video folder path (or press Enter for default '{VIDEO_FOLDER}'): ").strip()
    if not folder:
        folder = VIDEO_FOLDER

    videos = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ])

    if not videos:
        raise ValueError(f"No video files found in: {folder}")

    print(f"\n📂 Found {len(videos)} video(s):")
    for v in videos:
        print(f"  {v}")

    # ── 2. Concatenate with transitions ──────────────────────────────────────
    methods  = [transition_without_video, transition_with_video, simple_concat]
    weights  = [0.2, 0.4, 0.4]

    temp_dir = os.path.dirname(OUTPUT_NAME)
    os.makedirs(temp_dir, exist_ok=True)

    current_video = videos[0]

    for i in range(1, len(videos)):
        next_video   = videos[i]
        step_output  = os.path.join(temp_dir, f"_step_{i}.mp4")
        chosen       = random.choices(methods, weights=weights, k=1)[0]

        print(f"\n[{i}/{len(videos)-1}] {os.path.basename(current_video)}  +  {os.path.basename(next_video)}")
        print(f"  Method: {chosen.__name__}")
        chosen(current_video, next_video, step_output)
        current_video = step_output

    # ── 3. Theme detection ───────────────────────────────────────────────────
    print("\n🔍 Detecting video theme with CLIP…")
    final_clip = VideoFileClip(current_video)
    top_theme, top_score = analyze_theme(final_clip)
    print(f"  Top theme: {top_theme}  ({top_score:.2f})")

    # ── 4. Choose background music ───────────────────────────────────────────
    audio_path = choose_song(top_theme)

    # ── 5. Mix audio ─────────────────────────────────────────────────────────
    audio = AudioFileClip(audio_path)
    if audio.duration < final_clip.duration:
        audio = afx.audio_loop(audio, duration=final_clip.duration)
    else:
        audio = audio.subclip(0, final_clip.duration + 3)
    audio = audio.fx(afx.audio_fadein, 3).fx(afx.audio_fadeout, 3)

    # ── 6. Title card ────────────────────────────────────────────────────────
    title_txt = f"Detected theme: {top_theme} ({top_score:.2f})"
    w, h = final_clip.size

    try:
        title = (
            TextClip(title_txt, fontsize=48, font="DejaVu-Sans", color="white",
                     size=(w, None), method="caption")
            .set_duration(3)
            .on_color(size=(w, int(h * 0.15)), color=(0, 0, 0), col_opacity=0.6)
            .set_pos(("center", "center"))
        )
        final_with_title = concatenate_videoclips([title, final_clip], method="compose")
    except Exception as e:
        print(f"  ⚠️  TextClip failed ({e}) — skipping title card")
        final_with_title = final_clip

    final_with_title = final_with_title.set_audio(audio)

    # ── 7. Render ─────────────────────────────────────────────────────────────
    print(f"\n🎬 Rendering final video → {OUTPUT_NAME}")
    final_with_title.write_videofile(
        OUTPUT_NAME,
        codec="libx264",
        audio_codec="aac",
        fps=final_clip.fps or 24,
        threads=4,
        preset="medium",
    )

    print(f"\n✅ Done! Saved to: {OUTPUT_NAME}")

    # ── 8. Clean up temp files ────────────────────────────────────────────────
    for i in range(1, len(videos)):
        tmp = os.path.join(temp_dir, f"_step_{i}.mp4")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


if __name__ == "__main__":
    main()