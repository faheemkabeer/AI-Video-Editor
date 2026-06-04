#___________NOT FAST__________________________

from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, afx, TextClip, CompositeVideoClip
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image
import numpy as np
import os, random, torch
import moviepy.config as mp_config

!apt-get update
!apt-get install -y imagemagick
!sed -i '/<policy domain="path" rights="none"/d' /etc/ImageMagick-6/policy.xml
!sed -i 's#<policy domain="delegate" rights="none" stealth="true" id="default-delegate"/>#<policy domain="delegate" rights="read|write" stealth="true" id="default-delegate"/>#g' /etc/ImageMagick-6/policy.xml
mp_config.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

WEDDING_SONGS_DIR ="/content/drive/MyDrive/main/pattt/weddind"
NON_WEDDING_SONGS_DIR ="/content/drive/MyDrive/main/pattt/non weddind"

import cv2
import moviepy.video.fx.all as vfx

# =========================================================
# CUDA CHECK
# =========================================================
use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

if use_cuda:
    print("CUDA detected: GPU acceleration enabled")
else:
    print("CUDA not available: using CPU")

# =========================================================
# CUDA RESIZE WRAPPER
# =========================================================
def resize_frame(frame, width=None, height=None, fx=None, fy=None):
    if not use_cuda:
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

# =========================================================
# METHOD 1 : TRANSITION WITHOUT TRANSITION VIDEO (OpenCV)
# =========================================================
def transition_without_video(video1, video2, output):

    transition_frames = 20

    def zoom_crossfade(buffer1, buffer2, width, height):

        frames = []
        for i in range(len(buffer1)):

            frame1 = buffer1[i]
            frame2 = buffer2[i]

            scale = 1 + (i / len(buffer1)) * 0.5

            zoomed = resize_frame(frame1, fx=scale, fy=scale)

            h, w = zoomed.shape[:2]

            start_x = (w - width) // 2
            start_y = (h - height) // 2

            zoom_crop = zoomed[start_y:start_y+height, start_x:start_x+width]

            alpha = i / len(buffer1)

            frame = cv2.addWeighted(zoom_crop, 1-alpha, frame2, alpha, 0)

            frames.append(frame)

        return frames


    def zoom_in(buffer, width, height):

        frames = []

        for i in range(len(buffer)):

            frame = buffer[i]

            scale = 1 + (i / len(buffer)) * 2

            zoomed = resize_frame(frame, fx=scale, fy=scale)

            h, w = zoomed.shape[:2]

            start_x = (w - width) // 2
            start_y = (h - height) // 2

            crop = zoomed[start_y:start_y+height, start_x:start_x+width]

            frames.append(crop)

        return frames


    def dissolve(buffer1, buffer2):

        frames = []

        for i in range(len(buffer1)):

            alpha = i / len(buffer1)

            frame = cv2.addWeighted(buffer1[i], 1-alpha, buffer2[i], alpha, 0)

            frames.append(frame)

        return frames


    def shaky_zoom(buffer, width, height):

        frames = []

        for i in range(len(buffer)):

            frame = buffer[i]

            scale = 1 + (i / len(buffer))

            zoomed = resize_frame(frame, fx=scale, fy=scale)

            h, w = zoomed.shape[:2]

            shift_x = random.randint(-10, 10)
            shift_y = random.randint(-10, 10)

            start_x = (w - width)//2 + shift_x
            start_y = (h - height)//2 + shift_y

            start_x = max(0, min(start_x, w-width))
            start_y = max(0, min(start_y, h-height))

            crop = zoomed[start_y:start_y+height, start_x:start_x+width]

            frames.append(crop)

        return frames


    def pull_in(buffer, width, height):

        frames = []

        for i in range(len(buffer)):

            frame = buffer[i]

            scale = 1 + (i / len(buffer)) ** 2 * 3

            zoomed = resize_frame(frame, fx=scale, fy=scale)

            h, w = zoomed.shape[:2]

            start_x = (w - width) // 2
            start_y = (h - height) // 2

            crop = zoomed[start_y:start_y+height, start_x:start_x+width]

            frames.append(crop)

        return frames


    def pull_out(buffer, width, height):

        frames = []

        for i in range(len(buffer)):

            frame = buffer[i]

            progress = i / len(buffer)

            scale = 2 - progress

            zoomed = resize_frame(frame, fx=scale, fy=scale)

            h, w = zoomed.shape[:2]

            start_x = (w - width) // 2
            start_y = (h - height) // 2

            crop = zoomed[start_y:start_y+height, start_x:start_x+width]

            frames.append(crop)

        return frames


    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames1 - transition_frames):
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)

    buffer1 = []
    buffer2 = []

    for i in range(transition_frames):
        ret, frame = cap1.read()
        if ret:
            buffer1.append(frame)

    for i in range(transition_frames):
        ret, frame = cap2.read()
        if ret:
            frame = resize_frame(frame, width, height)
            buffer2.append(frame)

    transitions = [
        "zoom_crossfade",
        "zoom_in",
        "dissolve",
        "pull_in",
        "pull_out"
    ]

    chosen = random.choice(transitions)

    print("Selected Transition:", chosen)

    if chosen == "zoom_crossfade":
        frames = zoom_crossfade(buffer1, buffer2, width, height)

    elif chosen == "zoom_in":
        frames = zoom_in(buffer1, width, height)

    elif chosen == "dissolve":
        frames = dissolve(buffer1, buffer2)

    elif chosen == "pull_in":
        frames = pull_in(buffer1, width, height)

    elif chosen == "pull_out":
        frames = pull_out(buffer2, width, height)

    for f in frames:
        out.write(f)

    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame = resize_frame(frame, width, height)
        out.write(frame)

    cap1.release()
    cap2.release()
    out.release()


# =========================================================
# METHOD 2 : OVERLAY TRANSITION
# =========================================================
def transition_with_video(video1, video2, output):

    videoA = VideoFileClip(video1).without_audio()
    videoB = VideoFileClip(video2).without_audio()

    transition_folder = "/content/drive/MyDrive/main/transition"

    transition_files = [
        os.path.join(transition_folder, f)
        for f in os.listdir(transition_folder)
        if f.endswith((".mp4", ".mov", ".mkv"))
    ]

    transition_path = random.choice(transition_files)

    transition = VideoFileClip(transition_path)

    transition = transition.resize(videoA.size)
    videoB = videoB.resize(videoA.size)

    TRANSITION_DURATION = min(1.5, transition.duration)

    transition = transition.subclip(0, TRANSITION_DURATION)

    start = videoA.duration - TRANSITION_DURATION

    transition = transition.set_start(start)
    videoB = videoB.set_start(start).crossfadein(TRANSITION_DURATION)

    final = CompositeVideoClip([videoA, transition, videoB], size=videoA.size)

    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=videoA.fps)


# =========================================================
# METHOD 3 : SIMPLE CONCAT
# =========================================================
from moviepy.editor import VideoFileClip, concatenate_videoclips


def simple_concat(video1, video2, output):
    clip1 = None
    clip2 = None
    final_clip = None

    try:
        # Load clips
        clip1 = VideoFileClip(video1)
        clip2 = VideoFileClip(video2)

        # Ensure same resolution (important!)
        if clip1.size != clip2.size:
            clip2 = clip2.resize(clip1.size)

        # Ensure same FPS
        if clip1.fps != clip2.fps:
            clip2 = clip2.set_fps(clip1.fps)

        # Concatenate safely
        final_clip = concatenate_videoclips(
            [clip1, clip2],
            method="compose"  # handles different formats safely
        )

        # Export video
        final_clip.write_videofile(
            output,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            threads=4,
            preset="medium"
        )

        print("✅ Video concatenation completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Cleanup resources (VERY IMPORTANT)
        if clip1:
            clip1.close()
        if clip2:
            clip2.close()
        if final_clip:
            final_clip.close()
# =========================================================
# MULTI VIDEO PROCESSING
# =========================================================


# =========================================================
# LOAD ALL VIDEOS FROM A FOLDER
# =========================================================

'''video_folder = input("Enter video folder path: ").strip()

videos = sorted([
    os.path.join(video_folder, f)
    for f in os.listdir(video_folder)
    if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
])

if not videos:
    raise ValueError("No video files found!")

print("\nVideos Found:")
for v in videos:
    print(v)'''




from google.colab import files
import os

# =========================================================
# UPLOAD VIDEOS FROM LOCAL COMPUTER
# =========================================================

uploaded = files.upload()

videos = []

for filename in uploaded.keys():

    # Colab saves uploaded files in current directory
    full_path = os.path.abspath(filename)

    videos.append(full_path)

print("\nUploaded Videos:")
for v in videos:
    print(v)




'''videos = [
    "/content/drive/MyDrive/main/videos/c1.mp4",
    "/content/drive/MyDrive/main/videos/c2.mp4",
    "/content/drive/MyDrive/main/videos/c3.mp4",
    "/content/drive/MyDrive/main/videos/c4.mp4",
    "/content/drive/MyDrive/main/videos/c5.mp4",
    "/content/drive/MyDrive/main/videos/c6.mp4",
    "/content/drive/MyDrive/main/videos/c7.mp4",
]'''





methods = [transition_without_video, transition_with_video, simple_concat]
weights = [0.2, 0.4, 0.4]

current_video = videos[0]

for i in range(1, len(videos)):
    next_video = videos[i]
    output = f"/content/step_{i}.mp4"
    chosen_method = random.choices(methods, weights=weights, k=1)[0]
    print("Processing:", current_video, "+", next_video)
    print("Chosen Method:", chosen_method.__name__)
    chosen_method(current_video, next_video, output)
    current_video = output

# =========================================================
# LOAD FINAL CLIP
# =========================================================
OUTPUT_NAME="final.mp4"
SAMPLE_EVERY_SEC=1.5

final_clip = VideoFileClip(current_video)

device = "cuda" if torch.cuda.is_available() else "cpu"

THEMES = [
    "wedding ceremony","bride","groom","dance","kiss",
    "group of people","party","ring exchange","cake cutting",
    "speeches","couple portrait","crowd","outdoor","indoor",
    "dinner","reception","romantic","family","friends","music performance"
]

print("Loading pretrained CLIP model...")
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

timestamps = np.arange(0, final_clip.duration, SAMPLE_EVERY_SEC)
theme_scores = np.zeros(len(THEMES))

for t in tqdm(timestamps, desc="Analyzing frames"):
    frame = final_clip.get_frame(t)
    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    inputs = processor(text=THEMES, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        theme_scores += logits

theme_scores /= theme_scores.sum()
top_idx = int(np.argmax(theme_scores))
top_theme = THEMES[top_idx]
top_score = float(theme_scores[top_idx])
print(f"Top detected theme: {top_theme} ({top_score:.2f})")

# =========================================================
# 🎵 NEW: MULTIPLE SONG SUGGESTIONS + USER SELECTION
# =========================================================

from IPython.display import Audio, display

def pick_multiple_audios(folder_path, n=3):
    audios = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".mpeg", ".mpg"))
    ]
    random.shuffle(audios)
    return audios[:n]

WEDDING_KEYWORDS = ["wedding", "bride", "groom", "couple", "reception", "ceremony"]

if any(k in top_theme.lower() for k in WEDDING_KEYWORDS):
    candidate_songs = pick_multiple_audios(WEDDING_SONGS_DIR, 3)
else:
    candidate_songs = pick_multiple_audios(NON_WEDDING_SONGS_DIR, 3)

if not candidate_songs:
    raise ValueError("❌ No audio files found!")

print("\n🎵 Suggested Songs:")
for i, song in enumerate(candidate_songs):
    print(f"{i+1}. {os.path.basename(song)}")

from IPython.display import Audio, display
import tempfile

PREVIEW_DURATION = 10  # seconds

for i, song in enumerate(candidate_songs):
    print(f"\n▶ song {i+1}: {os.path.basename(song)}")

    preview_audio = AudioFileClip(song).subclip(0, PREVIEW_DURATION)

    temp_preview_path = f"/content/preview_{i}.mp3"

    preview_audio.write_audiofile(
        temp_preview_path,
        codec="libmp3lame",
        fps=44100,
        logger=None
    )

    display(Audio(temp_preview_path))

# =========================================================
# AUDIO PROCESSING (UNCHANGED)
# =========================================================
choice = int(input("\nEnter the number of the song you want to use: ")) - 1
if choice < 0 or choice >= len(candidate_songs):
    raise ValueError("Invalid selection!")

AUDIO_PATH = candidate_songs[choice]
print("✅ Selected:", os.path.basename(AUDIO_PATH))



audio = AudioFileClip(AUDIO_PATH)

if audio.duration < final_clip.duration:
    audio = afx.audio_loop(audio, duration=final_clip.duration)
else:
    audio = audio.subclip(0, final_clip.duration+3)

audio = audio.fx(afx.audio_fadein, 3).fx(afx.audio_fadeout, 3)

# =========================================================
# TITLE + RENDER
# =========================================================

title_txt = f"Detected theme: {top_theme} ({top_score:.2f})"
w, h = final_clip.size

title = (TextClip(title_txt, fontsize=48, font="DejaVu-Sans", color="white",
                  size=(w, None), method="caption")
         .set_duration(3)
         .on_color(size=(w, int(h*0.15)), color=(0,0,0), col_opacity=0.6)
         .set_pos(("center", "center")))

final_with_title = concatenate_videoclips(
    [title, final_clip], method="compose"
).set_audio(audio)

print("Rendering final video...")
final_with_title.write_videofile(
    OUTPUT_NAME,
    codec="libx264",
    audio_codec="aac",
    fps=final_clip.fps or 24
)

print("✅ Final video saved as:", OUTPUT_NAME)