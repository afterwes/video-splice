"""
video_splice.py — Randomly splice images and videos into a vertical video with audio.

This script takes a folder of media (images and videos), randomly selects and
arranges them into a single output video at 1080x1920 (portrait orientation).
An audio file is overlaid, and the final video length matches the audio duration.

Usage:
    python video_splice.py -i ./media -a ./music.mp3 -o output.mp4 --clip-length 3 --image-length 2
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# Output resolution — defaults to vertical/portrait, swapped by --landscape
# ---------------------------------------------------------------------------
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# File extensions we accept, stored as sets for fast lookup
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}


# ===========================================================================
# 1. FILE DISCOVERY
#    Walk the input folder and separate media files into images and videos.
# ===========================================================================

def discover_media_files(input_folder_path):
    """
    Scan the input folder for supported image and video files.

    Returns two lists:
        image_paths — list of Path objects pointing to image files
        video_paths — list of Path objects pointing to video files

    Raises SystemExit if no usable media is found.
    """
    image_paths = []
    video_paths = []
    images_filtered = 0
    videos_filtered = 0

    # Iterate over every file in the folder (non-recursive, top-level only)
    for entry in Path(input_folder_path).iterdir():
        if not entry.is_file():
            continue  # skip subdirectories

        # Lowercase the extension so ".JPG" and ".jpg" are treated the same
        file_extension = entry.suffix.lower()

        if file_extension in SUPPORTED_IMAGE_EXTENSIONS:
            from PIL import Image
            try:
                with Image.open(str(entry)) as img:
                    if img.width * img.height < 640000:
                        images_filtered += 1
                        continue
            except Exception:
                images_filtered += 1
                continue
            image_paths.append(entry)
        elif file_extension in SUPPORTED_VIDEO_EXTENSIONS:
            try:
                clip = VideoFileClip(str(entry), audio=False)
                w, h = clip.size
                clip.close()
                if w * h < 640000:
                    videos_filtered += 1
                    continue
            except Exception:
                videos_filtered += 1
                continue
            video_paths.append(entry)
        # Files with other extensions are silently ignored

    total_media_count = len(image_paths) + len(video_paths)
    if total_media_count == 0:
        print(f"ERROR: No supported media files found in '{input_folder_path}'.")
        print(f"  Supported images: {SUPPORTED_IMAGE_EXTENSIONS}")
        print(f"  Supported videos: {SUPPORTED_VIDEO_EXTENSIONS}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) and {len(video_paths)} video(s).")
    if images_filtered > 0:
        print(f"  Filtered out {images_filtered} image(s) below 640,000px quality threshold.")
    if videos_filtered > 0:
        print(f"  Filtered out {videos_filtered} video(s) below 640,000px quality threshold.")
    return image_paths, video_paths


# ===========================================================================
# 1b. SCENE DETECTION
#     Scan each video for cuts and build a map of single-shot segments.
# ===========================================================================

# Global shot map: { video_path_str: [(start, end), (start, end), ...] }
VIDEO_SHOT_MAP = {}


def detect_shots(video_paths):
    """
    Analyze each video for scene cuts using PySceneDetect's ContentDetector.
    Populates VIDEO_SHOT_MAP with a list of (start_sec, end_sec) tuples
    per video, where each tuple is a single continuous shot.
    """
    from scenedetect import open_video, SceneManager, ContentDetector

    print(f"\nAnalyzing {len(video_paths)} video(s) for scene cuts...")

    for video_path in video_paths:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if scene_list:
            shots = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
        else:
            clip = VideoFileClip(str(video_path), audio=False)
            shots = [(0.0, clip.duration)]
            clip.close()

        VIDEO_SHOT_MAP[str(video_path)] = shots
        print(f"  {video_path.name}: {len(shots)} shot(s)")

    print(f"  Shot analysis complete.")


def _expand_to_shots(video_paths):
    """
    Expand a list of video paths into a list of (path, start, end) shot
    tuples using VIDEO_SHOT_MAP. Each shot in a multi-shot video becomes
    its own pool entry so the same file can be used with different shots.
    """
    shots = []
    for vp in video_paths:
        shot_ranges = VIDEO_SHOT_MAP.get(str(vp))
        if shot_ranges:
            for start, end in shot_ranges:
                shots.append((vp, start, end))
        else:
            shots.append((vp, 0.0, None))
    return shots


# ===========================================================================
# 2. SCALING / RESIZING
#    Every clip must fill the 1080x1920 frame. We scale so the media covers
#    the full frame (no black bars), then center-crop any overflow.
# ===========================================================================

def resize_clip_to_fill_frame(clip):
    """
    Force-stretch a moviepy clip to exactly OUTPUT_WIDTH x OUTPUT_HEIGHT.
    Aspect ratio is not preserved — the clip will be warped to fit.
    """
    return clip.resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))



# ===========================================================================
# 3b. MULTI-TRANSIENT DETECTION
#     Detect kicks, snares, and hi-hats independently, then merge into a
#     unified transient map for sequence-based editing.
# ===========================================================================

TRANSIENT_BANDS = {
    "snare": (2000, 5000, 0.6),
    "hihat": (8000, 16000, 0.6),
}

MIN_TRANSIENT_GAP = 0.2
TRANSIENT_DELAY = 0.0


def _detect_kicks_beat_track(audio_samples, sample_rate):
    """
    Hybrid kick detection: find the first real kick via onset detection
    in the bass range, get the BPM via beat tracking, then build a
    regular grid of kicks starting from that first real hit.
    """
    # Step 1: Get BPM from beat tracking on the full signal
    tempo, _ = librosa.beat.beat_track(y=audio_samples, sr=sample_rate)
    bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    beat_interval = 60.0 / bpm

    # Step 2: Find the first real kick via onset detection on bass (30-120Hz)
    stft = librosa.stft(audio_samples)
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    bass_mask = (frequencies >= 30) & (frequencies <= 120)
    bass_stft = np.zeros_like(stft)
    bass_stft[bass_mask, :] = stft[bass_mask, :]
    bass_signal = librosa.istft(bass_stft)

    onset_envelope = librosa.onset.onset_strength(y=bass_signal, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        y=bass_signal, sr=sample_rate,
        onset_envelope=onset_envelope,
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    if len(onset_times) == 0:
        first_kick = 0.0
    else:
        first_kick = float(onset_times[0])

    # Step 3: Build a regular grid from the first kick using the BPM
    audio_duration = len(audio_samples) / sample_rate
    beat_times = []
    t = first_kick
    while t < audio_duration:
        beat_times.append(t)
        t += beat_interval

    return np.array(beat_times)


def _detect_transients_in_band(audio_samples, sample_rate, freq_low, freq_high,
                               min_gap_seconds=0.2, threshold_factor=0.6):
    """
    Generic transient detection within a frequency band.

    Isolates the given frequency range via STFT bandpass, computes the onset
    strength envelope, and returns timestamps of detected transients.
    """
    stft = librosa.stft(audio_samples)
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    band_mask = (frequencies >= freq_low) & (frequencies <= freq_high)
    band_stft = np.zeros_like(stft)
    band_stft[band_mask, :] = stft[band_mask, :]
    band_signal = librosa.istft(band_stft)

    onset_envelope = librosa.onset.onset_strength(y=band_signal, sr=sample_rate)
    threshold = np.mean(onset_envelope) + threshold_factor * np.std(onset_envelope)

    onset_frames = librosa.onset.onset_detect(
        y=band_signal, sr=sample_rate,
        onset_envelope=onset_envelope,
        backtrack=False,
        delta=threshold,
        wait=int(min_gap_seconds * sample_rate / 512),
    )
    return librosa.frames_to_time(onset_frames, sr=sample_rate)


def detect_all_transients(audio_file_path, kick_only=False):
    """
    Analyze audio and build a unified transient map.

    When kick_only is False, runs kick, snare, and hi-hat detection
    independently, then merges all detections into a single sorted
    timeline. When kick_only is True, only kick transients are used.

    Returns:
        transient_map  — sorted list of (timestamp, type_string) tuples.
                         Always starts with (0.0, "start") and ends with
                         (audio_duration, "end").
        audio_duration — length of the audio in seconds.
    """
    audio_samples, sample_rate = librosa.load(str(audio_file_path), sr=None, mono=True)
    audio_duration = librosa.get_duration(y=audio_samples, sr=sample_rate)

    raw_transients = []

    kick_times = _detect_kicks_beat_track(audio_samples, sample_rate)
    kick_times = kick_times + TRANSIENT_DELAY
    for t in kick_times:
        raw_transients.append((min(float(t), audio_duration), "kick"))

    if not kick_only:
        for transient_type, (freq_low, freq_high, threshold_factor) in TRANSIENT_BANDS.items():
            times = _detect_transients_in_band(
                audio_samples, sample_rate, freq_low, freq_high,
                min_gap_seconds=MIN_TRANSIENT_GAP,
                threshold_factor=threshold_factor,
            )
            times = times + TRANSIENT_DELAY
            for t in times:
                raw_transients.append((min(float(t), audio_duration), transient_type))

    raw_transients.sort(key=lambda x: x[0])

    # Enforce minimum gap — when two transients collide, keep the first
    filtered = []
    for timestamp, transient_type in raw_transients:
        if not filtered or (timestamp - filtered[-1][0]) >= MIN_TRANSIENT_GAP:
            filtered.append((timestamp, transient_type))

    # Bookend the timeline so sequences always have a start and end.
    # If the first transient is very close to 0, snap it to 0 instead
    # of inserting a separate "start" entry that creates a tiny gap.
    if filtered and filtered[0][0] <= 0.1:
        filtered[0] = (0.0, filtered[0][1])
    elif not filtered or filtered[0][0] > 0.1:
        filtered.insert(0, (0.0, "start"))
    if filtered[-1][0] < audio_duration - 0.01:
        filtered.append((audio_duration, "end"))

    return filtered, audio_duration


# ===========================================================================
# 4. CLIP BUILDERS
#    Functions to turn a single image or video file into a moviepy clip
#    that is correctly sized and trimmed to the requested duration.
# ===========================================================================

def build_image_clip(image_path, display_duration_seconds):
    """
    Create a moviepy clip from a static image.

    The image is shown for `display_duration_seconds`, scaled and cropped
    to fill the 1080x1920 frame.
    """
    # ImageClip loads the image and holds it for the given duration
    image_clip = ImageClip(str(image_path), duration=display_duration_seconds)

    # Scale and crop to fill the output frame
    fitted_clip = resize_clip_to_fill_frame(image_clip)

    return fitted_clip


def build_video_clip(shot_tuple, max_clip_duration_seconds):
    """
    Create a moviepy clip from a shot tuple (path, start, end), trimmed
    to the requested length within the shot's boundaries.
    """
    video_path, shot_start, shot_end = shot_tuple
    source_video = VideoFileClip(str(video_path), audio=False)

    if shot_end is not None:
        shot_end = min(shot_end, source_video.duration)
        shot_duration = shot_end - shot_start
        if shot_duration <= max_clip_duration_seconds:
            trimmed_clip = source_video.subclipped(shot_start, shot_end)
        else:
            latest_start = shot_end - max_clip_duration_seconds
            s = random.uniform(shot_start, latest_start)
            trimmed_clip = source_video.subclipped(s, s + max_clip_duration_seconds)
    else:
        source_duration = source_video.duration
        if source_duration <= max_clip_duration_seconds:
            trimmed_clip = source_video
        else:
            latest_start = source_duration - max_clip_duration_seconds
            s = random.uniform(0, latest_start)
            trimmed_clip = source_video.subclipped(s, s + max_clip_duration_seconds)

    fitted_clip = resize_clip_to_fill_frame(trimmed_clip)
    return fitted_clip


def _load_bg_video(shot_tuple, total_duration):
    """
    Load a background video for a sequence from a shot tuple (path, start, end).
    Uses the shot's time range directly. Loops if the shot is too short.
    """
    video_path, shot_start, shot_end = shot_tuple
    source_video = VideoFileClip(str(video_path), audio=False)

    if shot_end is not None:
        shot_end = min(shot_end, source_video.duration)
        shot_duration = shot_end - shot_start
        segment = source_video.subclipped(shot_start, shot_end)
        if shot_duration >= total_duration:
            latest_start = shot_start + (shot_duration - total_duration)
            s = random.uniform(shot_start, latest_start)
            bg = source_video.subclipped(s, s + total_duration)
        else:
            loops_needed = int(total_duration // segment.duration) + 1
            bg = concatenate_videoclips([segment] * loops_needed)
            bg = bg.subclipped(0, total_duration)
    elif source_video.duration >= total_duration:
        latest_start = source_video.duration - total_duration
        s = random.uniform(0, latest_start)
        bg = source_video.subclipped(s, s + total_duration)
    else:
        loops_needed = int(total_duration // source_video.duration) + 1
        bg = concatenate_videoclips([source_video] * loops_needed)
        bg = bg.subclipped(0, total_duration)
    return resize_clip_to_fill_frame(bg)


# ===========================================================================
# 4. SEQUENCE ASSEMBLY
#    Randomly pick media files and build clips until we've filled the
#    required total duration (which matches the audio length).
# ===========================================================================

def assemble_clip_sequence(
    image_paths,
    video_paths,
    target_total_duration_seconds,
    clip_duration_seconds,
    image_duration_seconds,
):
    """
    Build a list of moviepy clips that, when concatenated, fill the target duration.

    Uses fixed clip/image durations. Returns a list of moviepy clips ready
    for concatenation.
    """
    if not image_paths and not video_paths:
        print("ERROR: No media files provided. Cannot assemble sequence.")
        sys.exit(1)

    IMAGE_GROUP_SIZE = 10

    clip_sequence = []
    accumulated_duration = 0.0
    clip_index = 0
    unused_images = list(image_paths)
    unused_videos = _expand_to_shots(video_paths)
    random.shuffle(unused_images)
    random.shuffle(unused_videos)

    print(f"\nAssembling clips to fill {target_total_duration_seconds:.1f}s of audio...")

    def _refill_if_both_empty():
        nonlocal unused_images, unused_videos
        if not unused_images and not unused_videos:
            if image_paths:
                unused_images = list(image_paths)
                random.shuffle(unused_images)
            if video_paths:
                unused_videos = _expand_to_shots(video_paths)
                random.shuffle(unused_videos)
            print("  (all media used once — reshuffling)")

    while accumulated_duration < target_total_duration_seconds:
            remaining_duration = target_total_duration_seconds - accumulated_duration
            if remaining_duration < 0.1:
                break

            _refill_if_both_empty()

            eligible_types = []
            if unused_videos:
                eligible_types.append("video")
            if unused_images:
                eligible_types.append("image")

            chosen_type = random.choice(eligible_types)

            if chosen_type == "image":
                for _ in range(IMAGE_GROUP_SIZE):
                    remaining_duration = target_total_duration_seconds - accumulated_duration
                    if remaining_duration < 0.1:
                        break
                    _refill_if_both_empty()
                    if not unused_images:
                        break

                    image_path = unused_images.pop()
                    actual_duration = min(image_duration_seconds, remaining_duration)

                    clip_index += 1
                    print(f"  Clip {clip_index}: image — {image_path.name} ({actual_duration:.1f}s)")

                    new_clip = build_image_clip(image_path, actual_duration)
                    clip_sequence.append(new_clip)
                    accumulated_duration += new_clip.duration

                remaining_duration = target_total_duration_seconds - accumulated_duration
                if remaining_duration >= 0.1 and unused_videos:
                    shot = unused_videos.pop()
                    actual_duration = min(clip_duration_seconds, remaining_duration)

                    clip_index += 1
                    print(f"  Clip {clip_index}: video — {shot[0].name} ({actual_duration:.1f}s)")

                    new_clip = build_video_clip(shot, actual_duration)
                    clip_sequence.append(new_clip)
                    accumulated_duration += new_clip.duration
            else:
                shot = unused_videos.pop()
                actual_duration = min(clip_duration_seconds, remaining_duration)

                clip_index += 1
                print(f"  Clip {clip_index}: video — {shot[0].name} ({actual_duration:.1f}s)")

                new_clip = build_video_clip(shot, actual_duration)
                clip_sequence.append(new_clip)
                accumulated_duration += new_clip.duration

    print(f"\nTotal assembled duration: {accumulated_duration:.1f}s")
    return clip_sequence


# ===========================================================================
# 5. SEQUENCE-BASED EDITING
#    Each sequence type is a builder function that consumes a span of
#    transient timestamps and produces a single composite clip (which may
#    have layered video + image overlays). Random sequences are chained
#    together to fill the full audio duration.
# ===========================================================================


def _pop_image(image_paths, unused_images):
    if not unused_images:
        print("WARNING: All images used — repeating from the beginning. Add more images for variety.")
        unused_images.extend(image_paths)
        random.shuffle(unused_images)
    return unused_images.pop()


def _pop_landscape_image(image_paths, unused_images, allow_square=False, exclude=None):
    from PIL import Image
    excluded = set(str(p) for p in exclude) if exclude else set()
    for _ in range(len(unused_images)):
        candidate = unused_images.pop()
        if str(candidate) in excluded:
            unused_images.insert(0, candidate)
            continue
        with Image.open(str(candidate)) as img:
            if img.width > img.height or (allow_square and img.width == img.height):
                return candidate
        unused_images.insert(0, candidate)
    print("WARNING: Not enough landscape images — repeating. Add more landscape images for variety.")
    unused_images.extend(image_paths)
    random.shuffle(unused_images)
    for _ in range(len(unused_images)):
        candidate = unused_images.pop()
        if str(candidate) in excluded:
            unused_images.insert(0, candidate)
            continue
        with Image.open(str(candidate)) as img:
            if img.width > img.height or (allow_square and img.width == img.height):
                return candidate
        unused_images.insert(0, candidate)
    if unused_images:
        return unused_images.pop()
    return _pop_image(image_paths, unused_images)


def _pop_landscape_video(video_paths, unused_videos):
    for _ in range(len(unused_videos)):
        candidate = unused_videos.pop()
        clip = VideoFileClip(str(candidate[0]), audio=False)
        w, h = clip.size
        clip.close()
        if w > h:
            return candidate
        unused_videos.insert(0, candidate)
    print("WARNING: Not enough landscape videos — repeating. Add more landscape videos for variety.")
    unused_videos.extend(_expand_to_shots(video_paths))
    random.shuffle(unused_videos)
    for _ in range(len(unused_videos)):
        candidate = unused_videos.pop()
        clip = VideoFileClip(str(candidate[0]), audio=False)
        w, h = clip.size
        clip.close()
        if w > h:
            return candidate
        unused_videos.insert(0, candidate)
    if unused_videos:
        return unused_videos.pop()
    return _pop_video(video_paths, unused_videos)


def _pop_video(video_paths, unused_videos):
    if not unused_videos:
        shots = _expand_to_shots(video_paths)
        if not shots:
            return None
        print("WARNING: All video shots used — repeating from the beginning. Add more videos for variety.")
        unused_videos.extend(shots)
        random.shuffle(unused_videos)
    return unused_videos.pop()


def _pop_portrait_image(image_paths, unused_images):
    from PIL import Image
    for _ in range(len(unused_images)):
        candidate = unused_images.pop()
        with Image.open(str(candidate)) as img:
            if img.height > img.width:
                return candidate
        unused_images.insert(0, candidate)
    print("WARNING: Not enough portrait images — repeating. Add more portrait images for variety.")
    unused_images.extend(image_paths)
    random.shuffle(unused_images)
    for _ in range(len(unused_images)):
        candidate = unused_images.pop()
        with Image.open(str(candidate)) as img:
            if img.height > img.width:
                return candidate
        unused_images.insert(0, candidate)
    if unused_images:
        return unused_images.pop()
    return _pop_image(image_paths, unused_images)


def _pop_portrait_video(video_paths, unused_videos):
    for _ in range(len(unused_videos)):
        candidate = unused_videos.pop()
        clip = VideoFileClip(str(candidate[0]), audio=False)
        w, h = clip.size
        clip.close()
        if h > w:
            return candidate
        unused_videos.insert(0, candidate)
    print("WARNING: Not enough portrait videos — repeating. Add more portrait videos for variety.")
    unused_videos.extend(_expand_to_shots(video_paths))
    random.shuffle(unused_videos)
    for _ in range(len(unused_videos)):
        candidate = unused_videos.pop()
        clip = VideoFileClip(str(candidate[0]), audio=False)
        w, h = clip.size
        clip.close()
        if h > w:
            return candidate
        unused_videos.insert(0, candidate)
    if unused_videos:
        return unused_videos.pop()
    return _pop_video(video_paths, unused_videos)


def _pop_image_for_mode(image_paths, unused_images):
    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        return _pop_landscape_image(image_paths, unused_images)
    return _pop_portrait_image(image_paths, unused_images)


def _pop_video_for_mode(video_paths, unused_videos):
    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        return _pop_landscape_video(video_paths, unused_videos)
    return _pop_portrait_video(video_paths, unused_videos)


def _overlay_size(orig_w, orig_h):
    aspect = orig_h / orig_w
    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        h = OUTPUT_HEIGHT // 2
        w = int(h / aspect)
    else:
        w = OUTPUT_WIDTH // 2
        h = int(w * aspect)
    return w, h


# ---------------------------------------------------------------------------
# Sequence: picture_in_video
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): video plays alone
#   Phase 2 (t1→t2): image overlaid at 540px wide, centered, native aspect ratio
#   Phase 3 (t2→t3): image expands to 1080px wide, centered, native aspect ratio
# ---------------------------------------------------------------------------

def build_picture_in_video(timestamps, image_paths, video_paths,
                           unused_images, unused_videos):
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    image_path = _pop_image_for_mode(image_paths, unused_images)

    bg_video = _load_bg_video(video_path, total_duration)

    raw_image = ImageClip(str(image_path))
    orig_w, orig_h = raw_image.size
    aspect = orig_h / orig_w

    # Phase 2: half-size overlay
    small_w, small_h = _overlay_size(orig_w, orig_h)
    small_overlay = raw_image.resized((small_w, small_h))
    small_overlay = small_overlay.with_duration(t2 - t1).with_start(t1 - t0)
    small_overlay = small_overlay.with_position((
        (OUTPUT_WIDTH - small_w) // 2,
        (OUTPUT_HEIGHT - small_h) // 2,
    ))

    # Phase 3: full-frame overlay
    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        full_h = OUTPUT_HEIGHT
        full_w = int(full_h / aspect)
    else:
        full_w = OUTPUT_WIDTH
        full_h = int(full_w * aspect)
    full_overlay = raw_image.resized((full_w, full_h))
    full_overlay = full_overlay.with_duration(t3 - t2).with_start(t2 - t0)
    full_overlay = full_overlay.with_position((
        (OUTPUT_WIDTH - full_w) // 2,
        (OUTPUT_HEIGHT - full_h) // 2,
    ))

    composite = CompositeVideoClip(
        [bg_video, small_overlay, full_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: picture_in_picture
#   Same as picture_in_video but with an image background instead of video.
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): image 1 fullscreen
#   Phase 2 (t1→t2): image 2 overlaid at half-size, centered
#   Phase 3 (t2→t3): image 2 expands to full frame
# ---------------------------------------------------------------------------

def build_picture_in_picture(timestamps, image_paths, video_paths,
                             unused_images, unused_videos):
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    bg_path = _pop_image_for_mode(image_paths, unused_images)
    overlay_path = _pop_image_for_mode(image_paths, unused_images)

    bg = ImageClip(str(bg_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    bg = bg.with_duration(total_duration)

    raw_image = ImageClip(str(overlay_path))
    orig_w, orig_h = raw_image.size
    aspect = orig_h / orig_w

    small_w, small_h = _overlay_size(orig_w, orig_h)
    small_overlay = raw_image.resized((small_w, small_h))
    small_overlay = small_overlay.with_duration(t2 - t1).with_start(t1 - t0)
    small_overlay = small_overlay.with_position((
        (OUTPUT_WIDTH - small_w) // 2,
        (OUTPUT_HEIGHT - small_h) // 2,
    ))

    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        full_h = OUTPUT_HEIGHT
        full_w = int(full_h / aspect)
    else:
        full_w = OUTPUT_WIDTH
        full_h = int(full_w * aspect)
    full_overlay = raw_image.resized((full_w, full_h))
    full_overlay = full_overlay.with_duration(t3 - t2).with_start(t2 - t0)
    full_overlay = full_overlay.with_position((
        (OUTPUT_WIDTH - full_w) // 2,
        (OUTPUT_HEIGHT - full_h) // 2,
    ))

    composite = CompositeVideoClip(
        [bg, small_overlay, full_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: double_picture_in_picture
#   Same as double_picture_in_video but with an image background.
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): image 1 fullscreen
#   Phase 2 (t1→t2): image 2 overlaid at half-size, centered
#   Phase 3 (t2→t3): image 3 replaces image 2, same half-size
# ---------------------------------------------------------------------------

def build_double_picture_in_picture(timestamps, image_paths, video_paths,
                                    unused_images, unused_videos):
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    bg_path = _pop_image_for_mode(image_paths, unused_images)
    image1_path = _pop_image_for_mode(image_paths, unused_images)
    image2_path = _pop_image_for_mode(image_paths, unused_images)

    bg = ImageClip(str(bg_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    bg = bg.with_duration(total_duration)

    raw_img1 = ImageClip(str(image1_path))
    img1_w, img1_h = _overlay_size(*raw_img1.size)
    img1_overlay = raw_img1.resized((img1_w, img1_h))
    img1_overlay = img1_overlay.with_duration(t2 - t1).with_start(t1 - t0)
    img1_overlay = img1_overlay.with_position((
        (OUTPUT_WIDTH - img1_w) // 2,
        (OUTPUT_HEIGHT - img1_h) // 2,
    ))

    raw_img2 = ImageClip(str(image2_path))
    img2_w, img2_h = _overlay_size(*raw_img2.size)
    img2_overlay = raw_img2.resized((img2_w, img2_h))
    img2_overlay = img2_overlay.with_duration(t3 - t2).with_start(t2 - t0)
    img2_overlay = img2_overlay.with_position((
        (OUTPUT_WIDTH - img2_w) // 2,
        (OUTPUT_HEIGHT - img2_h) // 2,
    ))

    composite = CompositeVideoClip(
        [bg, img1_overlay, img2_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: double_picture_in_video
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): video plays alone
#   Phase 2 (t1→t2): image 1 overlaid at 540px wide, centered, native aspect ratio
#   Phase 3 (t2→t3): image 1 replaced by image 2 at 540px wide, centered
# ---------------------------------------------------------------------------

def build_double_picture_in_video(timestamps, image_paths, video_paths,
                                  unused_images, unused_videos):
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    image1_path = _pop_image_for_mode(image_paths, unused_images)
    image2_path = _pop_image_for_mode(image_paths, unused_images)

    bg_video = _load_bg_video(video_path, total_duration)

    # Phase 2: Image 1 at half-size
    raw_img1 = ImageClip(str(image1_path))
    img1_w, img1_h = _overlay_size(*raw_img1.size)
    img1_overlay = raw_img1.resized((img1_w, img1_h))
    img1_overlay = img1_overlay.with_duration(t2 - t1).with_start(t1 - t0)
    img1_overlay = img1_overlay.with_position((
        (OUTPUT_WIDTH - img1_w) // 2,
        (OUTPUT_HEIGHT - img1_h) // 2,
    ))

    # Phase 3: Image 2 replaces image 1, also half-size
    raw_img2 = ImageClip(str(image2_path))
    img2_w, img2_h = _overlay_size(*raw_img2.size)
    img2_overlay = raw_img2.resized((img2_w, img2_h))
    img2_overlay = img2_overlay.with_duration(t3 - t2).with_start(t2 - t0)
    img2_overlay = img2_overlay.with_position((
        (OUTPUT_WIDTH - img2_w) // 2,
        (OUTPUT_HEIGHT - img2_h) // 2,
    ))

    composite = CompositeVideoClip(
        [bg_video, img1_overlay, img2_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: bottom_top_picture
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): Image 1 fullscreen at 1080x1920
#   Phase 2 (t1→t2): Image 1 shifts up (centered at y=1440, screen y=480),
#                     cropped at midpoint. Image 2 shifts down (centered at
#                     y=480, screen y=1440), cropped at midpoint.
#   Phase 3 (t2→t3): Image 3 overlaid centered, native aspect ratio, 540px wide
# ---------------------------------------------------------------------------

def build_bottom_top_picture(timestamps, image_paths, video_paths,
                             unused_images, unused_videos):
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    image1_path = _pop_image_for_mode(image_paths, unused_images)
    image2_path = _pop_image_for_mode(image_paths, unused_images)
    image3_path = _pop_image_for_mode(image_paths, unused_images)

    # Phase 1: Image 1 fullscreen
    img1_full = ImageClip(str(image1_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    img1_full = img1_full.with_duration(t1 - t0).with_start(0)

    phase2_start = t1 - t0
    phase2_3_duration = t3 - t1

    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        # Landscape: left/right split on x-axis
        half_w = OUTPUT_WIDTH // 2
        quarter_w = OUTPUT_WIDTH // 4
        three_quarter_w = OUTPUT_WIDTH * 3 // 4

        img1_left = ImageClip(str(image1_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
        img1_left = img1_left.cropped(x1=quarter_w, x2=three_quarter_w)
        img1_left = img1_left.with_duration(phase2_3_duration).with_start(phase2_start)
        img1_left = img1_left.with_position((0, 0))

        img2_right = ImageClip(str(image2_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
        img2_right = img2_right.cropped(x1=quarter_w, x2=three_quarter_w)
        img2_right = img2_right.with_duration(phase2_3_duration).with_start(phase2_start)
        img2_right = img2_right.with_position((half_w, 0))

        img1_half = img1_left
        img2_half = img2_right
    else:
        # Portrait: top/bottom split on y-axis
        half_h = OUTPUT_HEIGHT // 2
        quarter_h = OUTPUT_HEIGHT // 4
        three_quarter_h = OUTPUT_HEIGHT * 3 // 4

        img1_top = ImageClip(str(image1_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
        img1_top = img1_top.cropped(y1=quarter_h, y2=three_quarter_h)
        img1_top = img1_top.with_duration(phase2_3_duration).with_start(phase2_start)
        img1_top = img1_top.with_position((0, 0))

        img2_bottom = ImageClip(str(image2_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
        img2_bottom = img2_bottom.cropped(y1=quarter_h, y2=three_quarter_h)
        img2_bottom = img2_bottom.with_duration(phase2_3_duration).with_start(phase2_start)
        img2_bottom = img2_bottom.with_position((0, half_h))

        img1_half = img1_top
        img2_half = img2_bottom

    raw_img3 = ImageClip(str(image3_path))
    overlay_w, overlay_h = _overlay_size(*raw_img3.size)
    img3_overlay = raw_img3.resized((overlay_w, overlay_h))
    img3_overlay = img3_overlay.with_duration(t3 - t2).with_start(t2 - t0)
    img3_overlay = img3_overlay.with_position((
        (OUTPUT_WIDTH - overlay_w) // 2,
        (OUTPUT_HEIGHT - overlay_h) // 2,
    ))

    composite = CompositeVideoClip(
        [img1_full, img1_half, img2_half, img3_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: video_halved
#   Consumes 2 transient gaps (3 timestamp points).
#   Phase 1 (t0→t1): video plays alone fullscreen
#   Phase 2 (t1→t2): video continues, image overlaid on left half.
#            Image is 1080x1920, centered horizontally at x=270,
#            cropped at x=540 so nothing spills past the midpoint.
# ---------------------------------------------------------------------------

def build_video_halved(timestamps, image_paths, video_paths,
                       unused_images, unused_videos):
    t0, t1, t2 = timestamps
    total_duration = t2 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    image_path = _pop_image_for_mode(image_paths, unused_images)

    bg_video = _load_bg_video(video_path, total_duration)

    raw_image = ImageClip(str(image_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        # Landscape: image on top half, cropped at y midpoint
        quarter_h = OUTPUT_HEIGHT // 4
        three_quarter_h = OUTPUT_HEIGHT * 3 // 4
        cropped_image = raw_image.cropped(y1=0, y2=three_quarter_h)
        cropped_image = cropped_image.with_duration(t2 - t1).with_start(t1 - t0)
        cropped_image = cropped_image.with_position((0, -quarter_h))
    else:
        # Portrait: image on left half, cropped at x midpoint
        quarter_w = OUTPUT_WIDTH // 4
        three_quarter_w = OUTPUT_WIDTH * 3 // 4
        cropped_image = raw_image.cropped(x1=0, x2=three_quarter_w)
        cropped_image = cropped_image.with_duration(t2 - t1).with_start(t1 - t0)
        cropped_image = cropped_image.with_position((-quarter_w, 0))

    composite = CompositeVideoClip(
        [bg_video, cropped_image],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: triple_image
#   Consumes 4 transient gaps (5 timestamp points).
#   Phase 1 (t0→t1): Image 1 fullscreen
#   Phase 2 (t1→t2): Image 2 overlaid in top third, half-width, native aspect
#   Phase 3 (t2→t3): Image 3 overlaid in middle third, half-width, native aspect
#   Phase 4 (t3→t4): Image 4 overlaid in bottom third, half-width, native aspect
# ---------------------------------------------------------------------------

def build_triple_image(timestamps, image_paths, video_paths,
                       unused_images, unused_videos):
    t0, t1, t2, t3, t4 = timestamps
    total_duration = t4 - t0

    img1_path = _pop_image_for_mode(image_paths, unused_images)
    used = [img1_path]
    img2_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)
    used.append(img2_path)
    img3_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)
    used.append(img3_path)
    img4_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)

    # Phase 1→4: Image 1 fullscreen background for the entire sequence
    bg = ImageClip(str(img1_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    bg = bg.with_duration(total_duration)

    raw2 = ImageClip(str(img2_path))
    raw3 = ImageClip(str(img3_path))
    raw4 = ImageClip(str(img4_path))
    w2, h2 = _overlay_size(*raw2.size)
    w3, h3 = _overlay_size(*raw3.size)
    w4, h4 = _overlay_size(*raw4.size)

    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        # Landscape: left/middle/right thirds on x-axis
        third_w = OUTPUT_WIDTH // 3

        img2_overlay = raw2.resized((w2, h2))
        img2_overlay = img2_overlay.with_duration(t4 - t1).with_start(t1 - t0)
        img2_overlay = img2_overlay.with_position((
            (third_w - w2) // 2,
            (OUTPUT_HEIGHT - h2) // 2,
        ))

        img3_overlay = raw3.resized((w3, h3))
        img3_overlay = img3_overlay.with_duration(t4 - t2).with_start(t2 - t0)
        img3_overlay = img3_overlay.with_position((
            third_w + (third_w - w3) // 2,
            (OUTPUT_HEIGHT - h3) // 2,
        ))

        img4_overlay = raw4.resized((w4, h4))
        img4_overlay = img4_overlay.with_duration(t4 - t3).with_start(t3 - t0)
        img4_overlay = img4_overlay.with_position((
            2 * third_w + (third_w - w4) // 2,
            (OUTPUT_HEIGHT - h4) // 2,
        ))
    else:
        # Portrait: top/middle/bottom thirds on y-axis
        third_h = OUTPUT_HEIGHT // 3

        img2_overlay = raw2.resized((w2, h2))
        img2_overlay = img2_overlay.with_duration(t4 - t1).with_start(t1 - t0)
        img2_overlay = img2_overlay.with_position((
            (OUTPUT_WIDTH - w2) // 2,
            (third_h - h2) // 2,
        ))

        img3_overlay = raw3.resized((w3, h3))
        img3_overlay = img3_overlay.with_duration(t4 - t2).with_start(t2 - t0)
        img3_overlay = img3_overlay.with_position((
            (OUTPUT_WIDTH - w3) // 2,
            third_h + (third_h - h3) // 2,
        ))

        img4_overlay = raw4.resized((w4, h4))
        img4_overlay = img4_overlay.with_duration(t4 - t3).with_start(t3 - t0)
        img4_overlay = img4_overlay.with_position((
            (OUTPUT_WIDTH - w4) // 2,
            2 * third_h + (third_h - h4) // 2,
        ))

    composite = CompositeVideoClip(
        [bg, img2_overlay, img3_overlay, img4_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: video_triple_image
#   Same as triple_image but with a video background instead of image 1.
#   Consumes 4 transient gaps (5 timestamp points).
#   Phase 1 (t0→t1): Video fullscreen
#   Phase 2 (t1→t2): Image 1 overlaid in top third, half-width, native aspect
#   Phase 3 (t2→t3): Image 2 overlaid in middle third, half-width, native aspect
#   Phase 4 (t3→t4): Image 3 overlaid in bottom third, half-width, native aspect
# ---------------------------------------------------------------------------

def build_video_triple_image(timestamps, image_paths, video_paths,
                             unused_images, unused_videos):
    t0, t1, t2, t3, t4 = timestamps
    total_duration = t4 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    used = []
    img2_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)
    used.append(img2_path)
    img3_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)
    used.append(img3_path)
    img4_path = _pop_landscape_image(image_paths, unused_images, allow_square=True, exclude=used)

    bg = _load_bg_video(video_path, total_duration)

    raw2 = ImageClip(str(img2_path))
    raw3 = ImageClip(str(img3_path))
    raw4 = ImageClip(str(img4_path))
    w2, h2 = _overlay_size(*raw2.size)
    w3, h3 = _overlay_size(*raw3.size)
    w4, h4 = _overlay_size(*raw4.size)

    if OUTPUT_WIDTH > OUTPUT_HEIGHT:
        third_w = OUTPUT_WIDTH // 3

        img2_overlay = raw2.resized((w2, h2))
        img2_overlay = img2_overlay.with_duration(t4 - t1).with_start(t1 - t0)
        img2_overlay = img2_overlay.with_position((
            (third_w - w2) // 2,
            (OUTPUT_HEIGHT - h2) // 2,
        ))

        img3_overlay = raw3.resized((w3, h3))
        img3_overlay = img3_overlay.with_duration(t4 - t2).with_start(t2 - t0)
        img3_overlay = img3_overlay.with_position((
            third_w + (third_w - w3) // 2,
            (OUTPUT_HEIGHT - h3) // 2,
        ))

        img4_overlay = raw4.resized((w4, h4))
        img4_overlay = img4_overlay.with_duration(t4 - t3).with_start(t3 - t0)
        img4_overlay = img4_overlay.with_position((
            2 * third_w + (third_w - w4) // 2,
            (OUTPUT_HEIGHT - h4) // 2,
        ))
    else:
        third_h = OUTPUT_HEIGHT // 3

        img2_overlay = raw2.resized((w2, h2))
        img2_overlay = img2_overlay.with_duration(t4 - t1).with_start(t1 - t0)
        img2_overlay = img2_overlay.with_position((
            (OUTPUT_WIDTH - w2) // 2,
            (third_h - h2) // 2,
        ))

        img3_overlay = raw3.resized((w3, h3))
        img3_overlay = img3_overlay.with_duration(t4 - t2).with_start(t2 - t0)
        img3_overlay = img3_overlay.with_position((
            (OUTPUT_WIDTH - w3) // 2,
            third_h + (third_h - h3) // 2,
        ))

        img4_overlay = raw4.resized((w4, h4))
        img4_overlay = img4_overlay.with_duration(t4 - t3).with_start(t3 - t0)
        img4_overlay = img4_overlay.with_position((
            (OUTPUT_WIDTH - w4) // 2,
            2 * third_h + (third_h - h4) // 2,
        ))

    composite = CompositeVideoClip(
        [bg, img2_overlay, img3_overlay, img4_overlay],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: image_cluster
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): fullscreen image
#   Phase 2 (t1→t3): images appear every 0.1s clustered around center at
#                     540px width with random offsets
# ---------------------------------------------------------------------------

def build_image_cluster(timestamps, image_paths, video_paths,
                        unused_images, unused_videos):
    import math
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    bg_path = _pop_image_for_mode(image_paths, unused_images)
    bg = build_image_clip(bg_path, total_duration)

    cluster_duration = t3 - t1
    cluster_start = t1 - t0
    num_overlays = 5
    interval = cluster_duration / num_overlays

    overlays = []
    angle_offset = random.uniform(0, 2 * math.pi)
    for i in range(num_overlays):
        img_path = _pop_image_for_mode(image_paths, unused_images)
        raw = ImageClip(str(img_path))
        orig_w, orig_h = raw.size
        overlay_w, overlay_h = _overlay_size(orig_w, orig_h)
        resized = raw.resized((overlay_w, overlay_h))

        appear_time = cluster_start + (i * interval)
        remaining = total_duration - appear_time
        if remaining <= 0:
            break
        resized = resized.with_duration(remaining).with_start(appear_time)

        angle = angle_offset + i * (2 * math.pi / num_overlays)
        radius = random.randint(60, 150)
        cx = (OUTPUT_WIDTH - overlay_w) // 2 + int(math.cos(angle) * radius)
        cy = (OUTPUT_HEIGHT - overlay_h) // 2 + int(math.sin(angle) * radius)
        resized = resized.with_position((cx, cy))
        overlays.append(resized)

    composite = CompositeVideoClip(
        [bg] + overlays,
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: video_cluster
#   Consumes 3 transient gaps (4 timestamp points).
#   Phase 1 (t0→t1): fullscreen video
#   Phase 2 (t1→t3): images appear every 0.1s clustered around center at
#                     540px width with random offsets
# ---------------------------------------------------------------------------

def build_video_cluster(timestamps, image_paths, video_paths,
                        unused_images, unused_videos):
    import math
    t0, t1, t2, t3 = timestamps
    total_duration = t3 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    bg = _load_bg_video(video_path, total_duration)

    cluster_duration = t3 - t1
    cluster_start = t1 - t0
    num_overlays = 5
    interval = cluster_duration / num_overlays

    overlays = []
    angle_offset = random.uniform(0, 2 * math.pi)
    for i in range(num_overlays):
        img_path = _pop_image_for_mode(image_paths, unused_images)
        raw = ImageClip(str(img_path))
        orig_w, orig_h = raw.size
        overlay_w, overlay_h = _overlay_size(orig_w, orig_h)
        resized = raw.resized((overlay_w, overlay_h))

        appear_time = cluster_start + (i * interval)
        remaining = total_duration - appear_time
        if remaining <= 0:
            break
        resized = resized.with_duration(remaining).with_start(appear_time)

        angle = angle_offset + i * (2 * math.pi / num_overlays)
        radius = random.randint(60, 150)
        cx = (OUTPUT_WIDTH - overlay_w) // 2 + int(math.cos(angle) * radius)
        cy = (OUTPUT_HEIGHT - overlay_h) // 2 + int(math.sin(angle) * radius)
        resized = resized.with_position((cx, cy))
        overlays.append(resized)

    composite = CompositeVideoClip(
        [bg] + overlays,
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence: four_quarters
#   Consumes 5 transient gaps (6 timestamp points).
#   Phase 1 (t0→t1): fullscreen image
#   Phase 2 (t1→t5): quarter image in top-left
#   Phase 3 (t2→t5): quarter image in bottom-right
#   Phase 4 (t3→t5): quarter image in top-right
#   Phase 5 (t4→t5): quarter image in bottom-left
# ---------------------------------------------------------------------------

def build_four_quarters(timestamps, image_paths, video_paths,
                        unused_images, unused_videos):
    t0, t1, t2, t3, t4, t5 = timestamps
    total_duration = t5 - t0

    bg_path = _pop_image_for_mode(image_paths, unused_images)
    bg = build_image_clip(bg_path, total_duration)

    quarter_w = OUTPUT_WIDTH // 2
    quarter_h = OUTPUT_HEIGHT // 2

    positions = [
        (0, 0),
        (quarter_w, quarter_h),
        (quarter_w, 0),
        (0, quarter_h),
    ]
    start_times = [t1, t2, t3, t4]

    overlays = []
    for pos, start in zip(positions, start_times):
        img_path = _pop_image_for_mode(image_paths, unused_images)
        raw = ImageClip(str(img_path))
        resized = raw.resized((quarter_w, quarter_h))
        resized = resized.with_duration(t5 - start).with_start(start - t0)
        resized = resized.with_position(pos)
        overlays.append(resized)

    composite = CompositeVideoClip(
        [bg] + overlays,
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(total_duration)

    return composite, unused_images, unused_videos


# ---------------------------------------------------------------------------
# Sequence registry
#   Each entry has:
#     name             — for logging
#     transition_points — how many timestamps the builder expects
#     min_gap          — minimum seconds between each transition point
#     build            — builder function
# ---------------------------------------------------------------------------

SEQUENCE_TYPES = [
    {
        "name": "picture_in_video",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "build": build_picture_in_video,
    },
    {
        "name": "picture_in_picture",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_picture_in_picture,
    },
    {
        "name": "double_picture_in_video",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "build": build_double_picture_in_video,
    },
    {
        "name": "double_picture_in_picture",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_double_picture_in_picture,
    },
    {
        "name": "bottom_top_picture",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_bottom_top_picture,
    },
    {
        "name": "video_halved",
        "transition_points": 3,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "build": build_video_halved,
    },
    {
        "name": "triple_image",
        "transition_points": 5,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_triple_image,
    },
    {
        "name": "video_triple_image",
        "transition_points": 5,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "build": build_video_triple_image,
    },
    {
        "name": "image_cluster",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_image_cluster,
    },
    {
        "name": "video_cluster",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "build": build_video_cluster,
    },
    {
        "name": "four_quarters",
        "transition_points": 6,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "build": build_four_quarters,
    },
]


def _pick_transition_points(transient_map, start_index, count, min_gap, trigger=None):
    """
    Starting at start_index, scan forward through the transient map to find
    `count` timestamps where each is at least `min_gap` seconds after the
    previous one.

    If trigger is set (e.g. "kick"), only transients of that type (plus the
    bookend "start"/"end" entries) are considered. This prevents sequences
    from consuming snare/hihat timestamps as structural transition points.

    Returns (timestamps, end_index). If there aren't enough qualifying
    points left in the map, timestamps will have fewer than `count` entries.
    """
    allowed = {trigger, "start", "end"} if trigger else None
    points = [transient_map[start_index][0]]
    end_index = start_index

    while len(points) < count and end_index < len(transient_map) - 1:
        end_index += 1
        candidate_time = transient_map[end_index][0]
        candidate_type = transient_map[end_index][1]

        if allowed and candidate_type not in allowed:
            continue

        if candidate_time - points[-1] >= min_gap:
            points.append(candidate_time)

    return points, end_index


def assemble_sequence_mode(image_paths, video_paths, transient_map, audio_duration):
    """
    Build the full clip list using sequence-based editing.

    Picks random sequences one at a time, scans the transient map for
    transition points that satisfy each sequence's min_gap requirement,
    then calls the builder. Continues until the transient map is exhausted.
    """
    if not SEQUENCE_TYPES:
        print("ERROR: No sequence types defined yet. Add entries to SEQUENCE_TYPES.")
        sys.exit(1)

    print(f"\nAssembling sequence-mode edit for {audio_duration:.1f}s...")

    unused_images = list(image_paths)
    unused_videos = _expand_to_shots(video_paths)
    random.shuffle(unused_images)
    random.shuffle(unused_videos)

    has_videos = len(video_paths) > 0

    all_clips = []
    transient_index = 0
    seq_num = 0
    remaining_sequences = []

    while transient_index < len(transient_map) - 1:
        if not remaining_sequences:
            remaining_sequences = [
                s for s in SEQUENCE_TYPES
                if has_videos or not s.get("requires_video", False)
            ]
            random.shuffle(remaining_sequences)
            if not remaining_sequences:
                print("ERROR: No usable sequence types (no videos available for video sequences).")
                break

        placed = False
        for i, sequence in enumerate(remaining_sequences):
            points, end_index = _pick_transition_points(
                transient_map, transient_index,
                sequence["transition_points"],
                sequence["min_gap"],
                trigger=sequence.get("trigger"),
            )
            if len(points) < sequence["transition_points"]:
                continue

            seq_num += 1
            span = points[-1] - points[0]
            points_str = " → ".join(f"{p:.2f}s" for p in points)
            print(f"  Sequence {seq_num}: {sequence['name']} ({span:.2f}s) [{points_str}]")

            clip, unused_images, unused_videos = sequence["build"](
                points, image_paths, video_paths,
                unused_images, unused_videos,
            )
            all_clips.append(clip)
            transient_index = end_index
            remaining_sequences.pop(i)
            placed = True
            break

        if not placed:
            break

    total_duration = sum(c.duration for c in all_clips) if all_clips else 0
    print(f"\nTotal assembled duration: {total_duration:.1f}s ({len(all_clips)} clips)")
    return all_clips


# ===========================================================================
# 6. FINAL VIDEO RENDER
#    Concatenate all clips, attach the audio, and write to disk.
# ===========================================================================

def render_final_video(clip_sequence, audio_file_path, output_file_path):
    """
    Concatenate the clip sequence, overlay the audio, and export the video.

    Steps:
        1. Concatenate all clips in order into one continuous video.
        2. Load the audio file.
        3. Attach the audio to the video (trimmed to video length if needed).
        4. Write the result to the output file with reasonable encoding settings.
    """
    print("\nConcatenating clips into final video...")
    # method="compose" handles clips of slightly different sizes gracefully
    concatenated_video = concatenate_videoclips(clip_sequence, method="compose")

    # Load the audio track
    print(f"Loading audio from: {audio_file_path}")
    audio_track = AudioFileClip(str(audio_file_path))

    # Attach the audio — set the video's audio to our track
    # If audio is slightly longer than video, subclip it to match
    final_video_with_audio = concatenated_video.with_audio(
        audio_track.subclipped(0, concatenated_video.duration)
    )

    # Write the output file
    print(f"Rendering to: {output_file_path}")
    print("This may take a while depending on the video length...\n")

    final_video_with_audio.write_videofile(
        str(output_file_path),
        fps=30,                  # 30 frames per second — smooth standard
        codec="libx264",         # H.264 — widely compatible video codec
        audio_codec="aac",       # AAC — widely compatible audio codec
        preset="medium",         # Encoding speed vs quality tradeoff
        threads=4,               # Use multiple CPU threads for faster encoding
    )

    # Clean up moviepy resources to free memory
    concatenated_video.close()
    audio_track.close()

    print(f"\nDone! Output saved to: {output_file_path}")


# ===========================================================================
# 7. PINTEREST DOWNLOAD
#    Download all media from a Pinterest board URL using gallery-dl.
# ===========================================================================

def _get_gallery_dl_root():
    if getattr(sys, "frozen", False):
        return Path.home() / "Documents" / "VideoSplice" / "gallery-dl" / "pinterest"
    return Path(__file__).resolve().parent / "gallery-dl" / "pinterest"

GALLERY_DL_ROOT = _get_gallery_dl_root()


def _gallery_dl_cmd():
    return os.environ.get("GALLERY_DL_BINARY", "gallery-dl")


def parse_pinterest_url(pinterest_url):
    """
    Extract the username and board name from a Pinterest URL, decoding
    any URL-encoded characters (e.g. %C3%A9 -> é, %E9%9B%A2 -> 離).
    """
    from urllib.parse import unquote
    parts = pinterest_url.rstrip("/").split("/")
    # URL format: https://www.pinterest.com/<username>/<board_name>/
    username = unquote(parts[-2])
    board_name = unquote(parts[-1])
    return username, board_name


def _normalize_board_name(name):
    """
    Reduce a board name to only lowercase alphanumeric characters for
    fuzzy matching. Strips all punctuation, whitespace, and special
    characters so URL slugs like 'red-green' match folder names like
    'red & green __'.
    """
    import re
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_board_folder(username, board_name):
    """
    Locate the gallery-dl download folder for a board. Tries an exact
    match first, then falls back to normalized fuzzy matching against
    all folders in the username directory.
    """
    base_path = GALLERY_DL_ROOT / username
    if not base_path.is_dir():
        return None

    # Try exact match first
    exact = base_path / board_name
    if exact.is_dir():
        return exact

    # Fall back to normalized comparison against all folders
    target = _normalize_board_name(board_name)
    for folder in base_path.iterdir():
        if folder.is_dir() and _normalize_board_name(folder.name) == target:
            return folder

    # Substring match — URL slug may be a shortened form of the folder name
    for folder in base_path.iterdir():
        normalized_folder = _normalize_board_name(folder.name)
        if folder.is_dir() and (target in normalized_folder or normalized_folder in target):
            return folder

    return None


def _count_remote_pins(pinterest_url):
    """
    Use gallery-dl --simulate to count how many pins exist on the board
    without downloading anything.  Returns the count, or None if the
    command fails.
    """
    result = subprocess.run(
        [_gallery_dl_cmd(), "--simulate", pinterest_url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.startswith("#"))


def download_pinterest_board(pinterest_url):
    """
    Download all images/videos from a Pinterest board using gallery-dl.
    Skips the download if the local folder already has the same number
    of files as the remote board.  Returns the path to the folder where
    the media was saved.
    """
    username, board_name = parse_pinterest_url(pinterest_url)

    print(f"\nPinterest board: {pinterest_url}")
    print(f"  Username: {username}")
    print(f"  Board name: {board_name}")

    board_folder = find_board_folder(username, board_name)
    local_count = len(list(board_folder.iterdir())) if board_folder else 0

    print(f"  Checking remote pin count...")
    remote_count = _count_remote_pins(pinterest_url)

    if remote_count is not None:
        print(f"  Remote pins: {remote_count}, Local files: {local_count}")

    if remote_count is not None and board_folder and local_count == remote_count:
        print(f"  Local folder is up to date — skipping download.")
        print(f"  Using: {board_folder}")
        return str(board_folder)

    print("  Running gallery-dl (this may take a while)...")

    files_before = set(board_folder.iterdir()) if board_folder else set()

    GALLERY_DL_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [_gallery_dl_cmd(), "-d", str(GALLERY_DL_ROOT.parent.parent), pinterest_url],
        capture_output=True,
        text=True,
    )

    board_folder = find_board_folder(username, board_name)
    files_after = set(board_folder.iterdir()) if board_folder else set()

    new_files = files_after - files_before
    total_files = len(files_after)

    print(f"\n  gallery-dl results:")
    print(f"    New files downloaded: {len(new_files)}")
    print(f"    Total files in board folder: {total_files}")

    if result.stderr.strip():
        error_lines = [
            line.strip() for line in result.stderr.strip().splitlines()
            if "error" in line.lower()
        ]
        if error_lines:
            print(f"    Errors ({len(error_lines)}):")
            for line in error_lines:
                print(f"      {line}")

    if not board_folder:
        base_path = GALLERY_DL_ROOT / username
        available = [f.name for f in base_path.iterdir() if f.is_dir()] if base_path.is_dir() else []
        print(f"ERROR: Board folder not found for '{board_name}' under '{username}'")
        print(f"  Available folders: {available}")
        sys.exit(1)

    if result.returncode != 0 and total_files == 0:
        print(f"\nERROR: gallery-dl failed and no files found in board folder.")
        sys.exit(1)

    print(f"  Downloaded to: {board_folder}")
    return str(board_folder)


# ===========================================================================
# 8. ARGUMENT PARSING
#    Define the CLI interface with argparse.
# ===========================================================================

def parse_command_line_arguments():
    """
    Parse and validate command-line arguments.

    Returns the parsed arguments namespace with:
        .input        — path to the media folder
        .audio        — path to the audio file
        .output       — path for the output video
        .clip_length  — duration (seconds) for each video clip
        .image_length — duration (seconds) for each image
    """
    parser = argparse.ArgumentParser(
        description=(
            "Splice together random images and videos into a vertical (1080x1920) "
            "video with an audio track. Output length matches the audio file."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "-i", "--input",
        help="Path to a local folder containing images (.jpg, .png) and videos (.mp4, .mov)",
    )

    input_group.add_argument(
        "-p", "--pinterest",
        help="URL of a Pinterest board to download and use as media source",
    )

    parser.add_argument(
        "-a", "--audio",
        required=True,
        help="Path to the audio file (.mp3 or .wav) to use as the soundtrack",
    )

    parser.add_argument(
        "-o", "--output-dir",
        default=str(Path.home() / "Desktop"),
        help="Directory for output video files (default: current directory)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of videos to generate (default: 1). Each gets a unique random sequence.",
    )

    parser.add_argument(
        "--clip-length",
        type=float,
        default=5.0,
        help="Duration in seconds for each video clip in the final video (default: 5.0)",
    )

    parser.add_argument(
        "--image-length",
        type=float,
        default=3.0,
        help="Duration in seconds each image is displayed in the final video (default: 3.0)",
    )

    media_type_group = parser.add_mutually_exclusive_group()

    media_type_group.add_argument(
        "--images-only",
        action="store_true",
        help="Only use images from the media source (ignore videos)",
    )

    media_type_group.add_argument(
        "--videos-only",
        action="store_true",
        help="Only use videos from the media source (ignore images)",
    )

    parser.add_argument(
        "--sequence-mode",
        action="store_true",
        help="Use sequence-based editing with kick detection. Best with four-on-the-floor audio.",
    )


    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Output in landscape (1920x1080) instead of portrait (1080x1920).",
    )

    arguments = parser.parse_args()

    # --- Validation ---

    # Check that the input folder exists and is a directory (only when using --input)
    if arguments.input and not os.path.isdir(arguments.input):
        parser.error(f"Input folder does not exist or is not a directory: {arguments.input}")

    # Check that the audio file exists
    if not os.path.isfile(arguments.audio):
        parser.error(f"Audio file does not exist: {arguments.audio}")

    # Check that the audio file has a supported extension
    audio_extension = Path(arguments.audio).suffix.lower()
    if audio_extension not in SUPPORTED_AUDIO_EXTENSIONS:
        parser.error(
            f"Unsupported audio format '{audio_extension}'. "
            f"Supported: {SUPPORTED_AUDIO_EXTENSIONS}"
        )

    # Validate output directory
    if not os.path.isdir(arguments.output_dir):
        parser.error(f"Output directory does not exist: {arguments.output_dir}")

    # Count must be at least 1
    if arguments.count < 1:
        parser.error("--count must be at least 1")

    if arguments.sequence_mode and (arguments.images_only or arguments.videos_only):
        parser.error("--images-only and --videos-only cannot be used with --sequence-mode")

    # Clip and image lengths must be positive
    if arguments.clip_length <= 0:
        parser.error("--clip-length must be a positive number")
    if arguments.image_length <= 0:
        parser.error("--image-length must be a positive number")

    return arguments


# ===========================================================================
# 7. MAIN — tie everything together
# ===========================================================================

def run_pipeline(arguments, progress_callback=None):
    """
    Run the full video splice pipeline with the given arguments object.
    Can be called from CLI parsing or from the GUI.

    progress_callback, if provided, is called with (current, total) ints
    at each major pipeline stage so the caller can update a progress bar.
    """
    def _progress(current, total):
        if progress_callback:
            progress_callback(current, total)

    global OUTPUT_WIDTH, OUTPUT_HEIGHT
    if getattr(arguments, "landscape", False):
        OUTPUT_WIDTH = 1920
        OUTPUT_HEIGHT = 1080

    # If a Pinterest URL was given, download the board once
    if arguments.pinterest:
        input_folder = download_pinterest_board(arguments.pinterest)
    else:
        input_folder = arguments.input

    # Build output filenames from the audio file name
    audio_stem = Path(arguments.audio).stem
    output_dir = arguments.output_dir

    print("=" * 60)
    print("  VIDEO SPLICE — Random Media Montage Generator")
    print("=" * 60)
    print(f"  Input source : {arguments.pinterest or input_folder}")
    print(f"  Audio file   : {arguments.audio}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Count        : {arguments.count}")
    if arguments.sequence_mode:
        print(f"  Mode         : sequence (kick only)")
    else:
        print(f"  Clip length  : {arguments.clip_length}s")
        print(f"  Image length : {arguments.image_length}s")
    media_filter = "images only" if arguments.images_only else "videos only" if arguments.videos_only else "all"
    print(f"  Media filter : {media_filter}")
    print(f"  Resolution   : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} ({'landscape' if getattr(arguments, 'landscape', False) else 'portrait'})")
    print("=" * 60)

    total_steps = 2 + arguments.count
    _progress(0, total_steps)

    # Discover media files once
    image_paths, video_paths = discover_media_files(input_folder)

    if arguments.images_only:
        video_paths = []
        if not image_paths:
            print("ERROR: --images-only was set but no images were found.")
            sys.exit(1)
        print(f"Using images only ({len(image_paths)} images).")
    elif arguments.videos_only:
        image_paths = []
        if not video_paths:
            print("ERROR: --videos-only was set but no videos were found.")
            sys.exit(1)
        print(f"Using videos only ({len(video_paths)} videos).")

    # Analyze videos for scene cuts
    if video_paths:
        detect_shots(video_paths)

    _progress(1, total_steps)

    # Load audio duration once
    audio_for_duration_check = AudioFileClip(str(arguments.audio))
    target_total_duration = audio_for_duration_check.duration
    audio_for_duration_check.close()

    print(f"\nAudio duration: {target_total_duration:.1f}s — this will be the output video length.")

    # Analyze audio once if sequence mode is enabled
    transient_map = None
    if arguments.sequence_mode:
        transient_map, _ = detect_all_transients(arguments.audio, kick_only=True)

    # Find the first available number to avoid overwriting existing files
    starting_number = 1
    while (Path(output_dir) / f"{audio_stem}_{starting_number}.mp4").exists():
        starting_number += 1

    # Generate each video
    for video_number in range(starting_number, starting_number + arguments.count):
        output_filename = f"{audio_stem}_{video_number}.mp4"
        output_path = str(Path(output_dir) / output_filename)

        print(f"\n{'=' * 60}")
        print(f"  Generating video {video_number}/{arguments.count}: {output_filename}")
        print(f"{'=' * 60}")

        if arguments.sequence_mode:
            clip_sequence = assemble_sequence_mode(
                image_paths=image_paths,
                video_paths=video_paths,
                transient_map=transient_map,
                audio_duration=target_total_duration,
            )
        else:
            clip_sequence = assemble_clip_sequence(
                image_paths=image_paths,
                video_paths=video_paths,
                target_total_duration_seconds=target_total_duration,
                clip_duration_seconds=arguments.clip_length,
                image_duration_seconds=arguments.image_length,
            )

        if not clip_sequence:
            print("ERROR: No clips were assembled. The audio may be too short "
                  "or beat detection found too few beats for any sequence.")
            sys.exit(1)

        render_final_video(
            clip_sequence=clip_sequence,
            audio_file_path=arguments.audio,
            output_file_path=output_path,
        )

        _progress(2 + (video_number - starting_number + 1), total_steps)

    print(f"\nAll done! Generated {arguments.count} video(s) in {output_dir}/")


def main():
    """CLI entry point — parse arguments and run the pipeline."""
    arguments = parse_command_line_arguments()
    run_pipeline(arguments)


# Standard Python idiom: only run main() when this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
