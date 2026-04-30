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
import tempfile
import time
from pathlib import Path

import shutil

import librosa
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
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
OUTPUT_FPS = 30

# File extensions we accept, stored as sets for fast lookup
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".gif"}
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
                    if img.width * img.height < 360000:
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
                if w * h < 360000:
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
        print(f"  Filtered out {images_filtered} image(s) below 360,000px quality threshold.")
    if videos_filtered > 0:
        print(f"  Filtered out {videos_filtered} video(s) below 360,000px quality threshold.")
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

    skipped = []
    for video_path in video_paths:
        try:
            video = open_video(str(video_path))
        except Exception:
            print(f"  {video_path.name}: skipped (could not open)")
            skipped.append(video_path)
            continue
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

    for path in skipped:
        video_paths.remove(path)

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

_SEGMENT_CACHE = {}

def _is_valid_segment(path):
    """Check that a temp segment file exists and has a readable video stream."""
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return False
    ffmpeg = _get_ffmpeg_path()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True, text=True,
    )
    return "Duration: N/A" not in probe.stderr and "Video:" in probe.stderr


def _extract_segment(video_path, start, end):
    """
    Extract a video segment via ffmpeg and re-encode to constant 30fps.
    Bypasses moviepy's seeking issues with VFR and B-frame sources.
    Cached by (path, start, end) so the same segment is never encoded twice.
    """
    key = (str(video_path), round(start, 3), round(end, 3) if end is not None else None)
    if key in _SEGMENT_CACHE:
        cached = _SEGMENT_CACHE[key]
        if _is_valid_segment(cached):
            return VideoFileClip(cached, audio=False)
        _SEGMENT_CACHE.pop(key)

    ffmpeg = _get_ffmpeg_path()
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"vs_seg_{hash(key) & 0xFFFFFFFF:08x}.mp4",
    )
    if os.path.isfile(tmp):
        os.remove(tmp)

    cmd = [ffmpeg, "-hide_banner", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-an", "-vf", f"fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-bf", "0", "-g", str(OUTPUT_FPS),
            "-movflags", "+faststart",
            tmp]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not _is_valid_segment(tmp):
        print(f"  WARNING: segment extraction failed for {video_path} "
              f"[{start:.3f}-{end:.3f}], falling back to moviepy")
        clip = VideoFileClip(str(video_path), audio=False)
        if end is not None:
            end = min(end, clip.duration)
        start = min(start, clip.duration - 0.1)
        if end is not None and end > start:
            clip = clip.subclipped(start, end)
        return clip

    _SEGMENT_CACHE[key] = tmp
    return VideoFileClip(tmp, audio=False)


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


def _detect_kicks_beat_track(audio_samples, sample_rate, user_bpm=None):
    """
    Hybrid kick detection: find the first real kick via onset detection
    in the bass range, get the BPM via beat tracking, then build a
    regular grid of kicks starting from that first real hit.
    """
    # Step 1: Get BPM — use the user-supplied value if provided
    if user_bpm:
        bpm = float(user_bpm)
        print(f"  Using user-supplied BPM: {bpm}")
    else:
        tempo, _ = librosa.beat.beat_track(y=audio_samples, sr=sample_rate)
        bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        if bpm < 80:
            print(f"  Detected BPM: {bpm:.1f} (half-time, doubling to {bpm * 2:.1f})")
            bpm *= 2
        else:
            print(f"  Detected BPM: {bpm:.1f}")
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


def detect_all_transients(audio_file_path, kick_only=False, user_bpm=None):
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

    kick_times = _detect_kicks_beat_track(audio_samples, sample_rate, user_bpm=user_bpm)
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
    Each segment is extracted via ffmpeg to guarantee constant 30fps.
    """
    video_path, shot_start, shot_end = shot_tuple

    if shot_end is not None:
        shot_duration = shot_end - shot_start
        if shot_duration > max_clip_duration_seconds:
            latest_start = shot_end - max_clip_duration_seconds
            shot_start = random.uniform(shot_start, latest_start)
            shot_end = shot_start + max_clip_duration_seconds
    else:
        probe = VideoFileClip(str(video_path), audio=False)
        dur = probe.duration
        probe.close()
        if dur > max_clip_duration_seconds:
            shot_start = random.uniform(0, dur - max_clip_duration_seconds)
            shot_end = shot_start + max_clip_duration_seconds
        else:
            shot_end = dur

    clip = _extract_segment(video_path, shot_start, shot_end)
    return resize_clip_to_fill_frame(clip)


def _load_bg_video(shot_tuple, total_duration):
    """
    Load a background video for a sequence from a shot tuple (path, start, end).
    Uses the shot's time range directly. Loops if the shot is too short.
    Each segment is extracted via ffmpeg to guarantee constant 30fps.
    """
    video_path, shot_start, shot_end = shot_tuple

    if shot_end is not None:
        shot_duration = shot_end - shot_start
        if shot_duration >= total_duration:
            latest_start = shot_start + (shot_duration - total_duration)
            s = random.uniform(shot_start, latest_start)
            bg = _extract_segment(video_path, s, s + total_duration)
        else:
            segment = _extract_segment(video_path, shot_start, shot_end)
            seg_path = _SEGMENT_CACHE.get(
                (str(video_path), round(shot_start, 3), round(shot_end, 3)))
            loops_needed = int(total_duration // segment.duration) + 1
            loop_clips = [segment] + [
                VideoFileClip(seg_path, audio=False) for _ in range(loops_needed - 1)
            ]
            bg = concatenate_videoclips(loop_clips, method="chain")
            bg = bg.subclipped(0, total_duration)
    else:
        probe = VideoFileClip(str(video_path), audio=False)
        dur = probe.duration
        probe.close()
        if dur >= total_duration:
            s = random.uniform(0, dur - total_duration)
            bg = _extract_segment(video_path, s, s + total_duration)
        else:
            segment = _extract_segment(video_path, 0, dur)
            seg_path = _SEGMENT_CACHE.get(
                (str(video_path), 0.0, round(dur, 3)))
            loops_needed = int(total_duration // segment.duration) + 1
            loop_clips = [segment] + [
                VideoFileClip(seg_path, audio=False) for _ in range(loops_needed - 1)
            ]
            bg = concatenate_videoclips(loop_clips, method="chain")
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
        img1_top = img1_top.cropped(y1=0, y2=half_h)
        img1_top = img1_top.with_duration(phase2_3_duration).with_start(phase2_start)
        img1_top = img1_top.with_position((0, 0))

        img2_bottom = ImageClip(str(image2_path)).resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
        img2_bottom = img2_bottom.cropped(y1=0, y2=half_h)
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
    num_overlays = 4
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
    num_overlays = 4
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


# Sequence: video_four_quarters
#   Consumes 5 transient gaps (6 timestamp points).
#   Same as four_quarters but starts with a fullscreen video instead of image.
# ---------------------------------------------------------------------------

def build_video_four_quarters(timestamps, image_paths, video_paths,
                              unused_images, unused_videos):
    t0, t1, t2, t3, t4, t5 = timestamps
    total_duration = t5 - t0

    video_path = _pop_video_for_mode(video_paths, unused_videos)
    bg = _load_bg_video(video_path, total_duration)

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
    # {
    #     "name": "bottom_top_picture",
    #     "transition_points": 4,
    #     "min_gap": 0,
    #     "trigger": "kick",
    #     "requires_video": False,
    #     "build": build_bottom_top_picture,
    # },
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
        "group": "cluster",
        "build": build_image_cluster,
    },
    {
        "name": "video_cluster",
        "transition_points": 4,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "group": "cluster",
        "build": build_video_cluster,
    },
    {
        "name": "four_quarters",
        "transition_points": 6,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": False,
        "group": "quarters",
        "build": build_four_quarters,
    },
    {
        "name": "video_four_quarters",
        "transition_points": 6,
        "min_gap": 0,
        "trigger": "kick",
        "requires_video": True,
        "group": "quarters",
        "build": build_video_four_quarters,
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


def assemble_sequence_mode(image_paths, video_paths, transient_map, audio_duration,
                           intro_offset=0.0):
    """
    Build the full clip list using sequence-based editing.

    Picks random sequences one at a time, scans the transient map for
    transition points that satisfy each sequence's min_gap requirement,
    then calls the builder. Continues until the transient map is exhausted.

    If intro_offset > 0, a black placeholder clip is prepended for that
    duration (hidden behind the intro overlay in the final render).
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
    used_groups = set()

    while transient_index < len(transient_map) - 1:
        if not remaining_sequences:
            remaining_sequences = [
                s for s in SEQUENCE_TYPES
                if (has_videos or not s.get("requires_video", False))
                and (s.get("group") is None or s["group"] not in used_groups)
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
            if sequence.get("group"):
                used_groups.add(sequence["group"])
            remaining_sequences.pop(i)
            placed = True
            break

        if not placed:
            break

    if intro_offset > 0:
        placeholder = ColorClip(
            size=(OUTPUT_WIDTH, OUTPUT_HEIGHT), color=(0, 0, 0),
        ).with_duration(intro_offset)
        all_clips.insert(0, placeholder)
        print(f"  Prepended {intro_offset:.1f}s black placeholder for intro overlap")

    total_duration = sum(c.duration for c in all_clips) if all_clips else 0
    print(f"\nTotal assembled duration: {total_duration:.1f}s ({len(all_clips)} clips)")
    return all_clips


# ===========================================================================
# 6. FINAL VIDEO RENDER
#    Concatenate all clips, attach the audio, and write to disk.
# ===========================================================================

def _get_ffmpeg_path():
    """Return the ffmpeg binary path, checking all known locations."""
    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _detect_hw_encoder():
    """
    Probe ffmpeg for hardware-accelerated H.264 encoders.

    Returns (codec, ffmpeg_params) for the best available encoder,
    or ("libx264", []) as a CPU fallback.
    """
    ffmpeg = _get_ffmpeg_path()

    # Ordered by preference: macOS VideoToolbox, NVIDIA NVENC, Intel QSV
    candidates = [
        ("h264_videotoolbox", ["-b:v", "15M"]),
        ("h264_nvenc",        ["-preset", "p4", "-cq", "23"]),
        ("h264_qsv",         ["-global_quality", "23"]),
    ]

    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        encoder_list = result.stdout
    except Exception:
        return "libx264", []

    for codec, params in candidates:
        if codec in encoder_list:
            # Verify the encoder actually works with a tiny test encode
            try:
                test = subprocess.run(
                    [ffmpeg, "-hide_banner", "-f", "lavfi", "-i",
                     "color=black:s=64x64:d=0.1", "-c:v", codec,
                     "-f", "null", "-"],
                    capture_output=True, timeout=10,
                )
                if test.returncode == 0:
                    return codec, params
            except Exception:
                continue

    return "libx264", []


def render_final_video(clip_sequence, audio_file_path, output_file_path,
                       intro_clip=None, intro_overlap=False, duck_db=-8.0,
                       render_progress_callback=None):
    """
    Concatenate the clip sequence, overlay the audio, and export the video.

    If intro_clip is provided and intro_overlap is False, it is prepended
    with its own audio before the main sequence.

    If intro_overlap is True, the intro clip plays over the beginning of the
    main sequence. The input audio is ducked by duck_db dB during the intro,
    then restored to full volume. The intro clip's own audio is mixed on top.
    """
    print("\nConcatenating clips into final video...")
    for clip in clip_sequence:
        nframes = round(clip.duration * OUTPUT_FPS)
        clip.duration = nframes / OUTPUT_FPS
    concatenated_video = concatenate_videoclips(clip_sequence, method="chain")
    safety_bg = ColorClip(
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT), color=(0, 0, 0),
    ).with_duration(concatenated_video.duration)
    concatenated_video = CompositeVideoClip(
        [safety_bg, concatenated_video],
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    ).with_duration(concatenated_video.duration)

    print(f"Loading audio from: {audio_file_path}")
    audio_track = AudioFileClip(str(audio_file_path))

    if intro_clip and intro_overlap:
        intro_dur = intro_clip.duration
        print(f"Overlapping intro clip ({intro_dur:.1f}s) with ducked audio ({duck_db} dB)...")

        duck_db = -abs(duck_db)
        duck_factor = 10 ** (duck_db / 20.0)
        ducked_section = audio_track.subclipped(0, intro_dur).with_volume_scaled(duck_factor)
        full_section = audio_track.subclipped(intro_dur, concatenated_video.duration)

        from moviepy import concatenate_audioclips
        main_audio = concatenate_audioclips([ducked_section, full_section])

        intro_boost = 10 ** (10.0 / 20.0)
        audio_layers = [main_audio]
        if intro_clip.audio is not None:
            audio_layers.append(
                intro_clip.audio.subclipped(0, intro_dur).with_volume_scaled(intro_boost)
            )
        mixed_audio = CompositeAudioClip(audio_layers)

        intro_video_only = intro_clip.without_audio()
        overlay = CompositeVideoClip(
            [concatenated_video, intro_video_only.with_position((0, 0))],
            size=(concatenated_video.w, concatenated_video.h),
        ).with_duration(concatenated_video.duration)

        final_video_with_audio = overlay.with_audio(mixed_audio)
    elif intro_clip:
        final_video_with_audio = concatenated_video.with_audio(
            audio_track.subclipped(0, concatenated_video.duration)
        )
        print(f"Prepending intro clip ({intro_clip.duration:.1f}s)...")
        intro_boost = 10 ** (10.0 / 20.0)
        if intro_clip.audio is not None:
            intro_clip = intro_clip.with_audio(
                intro_clip.audio.with_volume_scaled(intro_boost)
            )
        final_video_with_audio = concatenate_videoclips(
            [intro_clip, final_video_with_audio], method="compose"
        )
    else:
        final_video_with_audio = concatenated_video.with_audio(
            audio_track.subclipped(0, concatenated_video.duration)
        )

    def _add_noise(frame):
        noise = np.random.randint(-2, 3, frame.shape, dtype=np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    final_video_with_audio = final_video_with_audio.image_transform(_add_noise)

    # Write the output file
    print(f"Rendering to: {output_file_path}")

    codec, hw_params = _detect_hw_encoder()
    if codec != "libx264":
        print(f"Using GPU-accelerated encoder: {codec}")
    else:
        print("No GPU encoder available — using libx264 (CPU)")

    write_kwargs = dict(
        fps=OUTPUT_FPS,
        codec=codec,
        audio_codec="aac",
        threads=4,
    )

    extra_params = ["-bf", "0"]
    if codec == "libx264":
        write_kwargs["preset"] = "medium"
    if hw_params:
        extra_params += hw_params
    write_kwargs["ffmpeg_params"] = extra_params

    logger = "bar"
    if render_progress_callback:
        from proglog import ProgressBarLogger

        class _RenderLogger(ProgressBarLogger):
            def __init__(self):
                super().__init__()
                self._last_logged_pct = -1
                self._pass = 0

            def bars_callback(self, bar, attr, value, old_value=None):
                if attr == "total" and old_value is None:
                    self._pass += 1
                if attr != "index":
                    return
                total = self.bars.get(bar, {}).get("total", 0)
                if total <= 0:
                    return
                frac = value / total
                if self._pass <= 1:
                    render_progress_callback(frac * 0.4, 1.0)
                    pct = int(frac * 100)
                    if pct >= self._last_logged_pct + 10:
                        self._last_logged_pct = pct
                        print(f"  Rendering video: {pct}% ({value}/{total} frames)")
                else:
                    render_progress_callback(0.4 + frac * 0.6, 1.0)

            def callback(self, **changes):
                for key, val in changes.items():
                    if isinstance(val, str) and val.strip():
                        print(val)

        logger = _RenderLogger()

    render_start = time.time()
    final_video_with_audio.write_videofile(str(output_file_path), logger=logger, **write_kwargs)
    render_elapsed = time.time() - render_start

    # Clean up moviepy resources to free memory
    concatenated_video.close()
    audio_track.close()

    print(f"\nDone! Output saved to: {output_file_path}")
    print(f"Render time: {render_elapsed:.1f}s")


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
    Extract the username, board name, and optional section from a Pinterest URL.
    Decodes URL-encoded characters (e.g. %C3%A9 -> é, %E9%9B%A2 -> 離).

    Supports:
        https://www.pinterest.com/<username>/<board>/
        https://www.pinterest.com/<username>/<board>/<section>/
    """
    from urllib.parse import unquote, urlparse
    path_parts = [p for p in urlparse(pinterest_url).path.strip("/").split("/") if p]
    if len(path_parts) >= 3:
        username = unquote(path_parts[0])
        board_name = unquote(path_parts[1])
        section = unquote(path_parts[2])
        return username, board_name, section
    elif len(path_parts) == 2:
        username = unquote(path_parts[0])
        board_name = unquote(path_parts[1])
        return username, board_name, None
    else:
        return unquote(path_parts[-1]) if path_parts else "", "", None


def _normalize_board_name(name):
    """
    Reduce a board name to only lowercase alphanumeric characters for
    fuzzy matching. Strips all punctuation, whitespace, and special
    characters so URL slugs like 'red-green' match folder names like
    'red & green __'.
    """
    import re
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_board_folder(username, board_name, section=None):
    """
    Locate the gallery-dl download folder for a board (or board section).
    Tries an exact match first, then falls back to normalized fuzzy matching.

    For sectioned boards, gallery-dl creates:
        pinterest/<username>/<board>/<section>/
    """
    base_path = GALLERY_DL_ROOT / username
    if not base_path.is_dir():
        return None

    def _find_in(parent, name):
        exact = parent / name
        if exact.is_dir():
            return exact
        target = _normalize_board_name(name)
        for folder in parent.iterdir():
            if folder.is_dir() and _normalize_board_name(folder.name) == target:
                return folder
        for folder in parent.iterdir():
            normalized_folder = _normalize_board_name(folder.name)
            if folder.is_dir() and (target in normalized_folder or normalized_folder in target):
                return folder
        return None

    board_path = _find_in(base_path, board_name)
    if board_path is None:
        return None

    if section:
        return _find_in(board_path, section)

    return board_path


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


def download_pinterest_board(pinterest_url, progress_callback=None, force_refresh=False):
    """
    Download all images/videos from a Pinterest board using gallery-dl.
    Skips the download if the local folder already has the same number
    of files as the remote board.  Returns the path to the folder where
    the media was saved.
    """
    username, board_name, section = parse_pinterest_url(pinterest_url)

    print(f"\nPinterest board: {pinterest_url}")
    print(f"  Username: {username}")
    print(f"  Board name: {board_name}")
    if section:
        print(f"  Section: {section}")

    board_folder = find_board_folder(username, board_name, section)
    local_count = len(list(board_folder.iterdir())) if board_folder else 0

    print(f"  Checking remote pin count...")
    remote_count = _count_remote_pins(pinterest_url)

    if remote_count is not None:
        print(f"  Remote pins: {remote_count}, Local files: {local_count}")

    if remote_count is not None and board_folder and local_count == remote_count and not force_refresh:
        print(f"  Local folder is up to date — skipping download.")
        print(f"  Using: {board_folder}")
        return str(board_folder)

    if force_refresh and board_folder and board_folder.exists():
        print(f"  Force refresh enabled — clearing local folder and re-downloading.")
        shutil.rmtree(board_folder, ignore_errors=True)
        board_folder = None
        local_count = 0

    print("  Running gallery-dl (this may take a while)...")
    if progress_callback:
        progress_callback(1, 100)

    files_before = set(board_folder.iterdir()) if board_folder else set()

    GALLERY_DL_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [_gallery_dl_cmd(), "-d", str(GALLERY_DL_ROOT.parent), pinterest_url],
        capture_output=True,
        text=True,
    )

    board_folder = find_board_folder(username, board_name, section)
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
        board_path = base_path / board_name if base_path.is_dir() else base_path
        search_path = board_path if section and board_path.is_dir() else base_path
        available = [f.name for f in search_path.iterdir() if f.is_dir()] if search_path.is_dir() else []
        label = f"'{section}' under '{username}/{board_name}'" if section else f"'{board_name}' under '{username}'"
        print(f"ERROR: Board folder not found for {label}")
        print(f"  Available folders: {available}")
        sys.exit(1)

    if result.returncode != 0 and total_files == 0:
        print(f"\nERROR: gallery-dl failed and no files found in board folder.")
        sys.exit(1)

    print(f"  Downloaded to: {board_folder}")
    return str(board_folder)


# ===========================================================================
# 8. YOUTUBE INTRO
#    Download a clip from YouTube and prepare it as an intro.
# ===========================================================================

def _yt_dlp_cmd():
    return os.environ.get("YT_DLP_BINARY", "yt-dlp")


def _parse_timestamp(ts):
    """Parse a timestamp string into seconds. Accepts 'SS', 'M:SS', or 'H:MM:SS'."""
    parts = ts.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {ts}")


def is_yarn_url(url):
    """Check if a URL is a yarn.co / getyarn.io clip URL."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    return host in ("yarn.co", "www.yarn.co", "getyarn.io", "www.getyarn.io")


def download_yarn_clip(url):
    """Download a video clip from yarn.co / getyarn.io. Returns the file path."""
    import re
    import tempfile
    from urllib.parse import urlparse

    import requests

    cache_dir = Path(tempfile.gettempdir()) / "videosplice_intro"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for f in cache_dir.iterdir():
        f.unlink()

    path = urlparse(url).path
    match = re.search(r'yarn-clip/([a-f0-9-]+)', path)
    if not match:
        match = re.search(r'/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', path)
    if not match:
        print(f"ERROR: Could not extract clip ID from Yarn URL: {url}")
        sys.exit(1)

    clip_id = match.group(1)
    clip_page_url = f"https://getyarn.io/yarn-clip/{clip_id}"
    mp4_url = f"https://y.yarn.co/{clip_id}.mp4"
    output_path = cache_dir / "intro.mp4"

    print(f"\nDownloading Yarn clip: {clip_id}")
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })
        session.get(clip_page_url)
        session.headers.update({"Referer": "https://getyarn.io/"})
        resp = session.get(mp4_url, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        print(f"ERROR: Failed to download Yarn clip: {e}")
        sys.exit(1)

    print(f"  Downloaded to: {output_path}")
    return str(output_path)


def download_intro_video(url):
    """Download an intro video from YouTube or Yarn. Returns the file path."""
    if is_yarn_url(url):
        return download_yarn_clip(url)
    return download_youtube_video(url)


def download_youtube_video(url):
    """Download a YouTube video at the highest quality. Returns the file path."""
    import glob
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "videosplice_intro"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous downloads
    for f in cache_dir.iterdir():
        f.unlink()

    output_template = str(cache_dir / "intro.%(ext)s")

    ffmpeg_path = _get_ffmpeg_path()

    cmd = [
        _yt_dlp_cmd(),
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--ffmpeg-location", ffmpeg_path,
        "-o", output_template,
        "--no-playlist",
        url,
    ]

    print(f"\nDownloading intro video from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("ERROR: Failed to download YouTube video.")
        sys.exit(1)

    # Find whatever file yt-dlp produced
    downloaded = list(cache_dir.iterdir())
    if not downloaded:
        print("ERROR: Download completed but no file found.")
        sys.exit(1)

    output_path = downloaded[0]
    print(f"  Downloaded to: {output_path}")
    return str(output_path)


def prepare_intro_clip(video_path, start_seconds, end_seconds=None, caption_words=None):
    """Load a video, trim to start/end, scale width to OUTPUT_WIDTH preserving aspect ratio,
    and center on a black background at OUTPUT_WIDTH x OUTPUT_HEIGHT."""
    clip = VideoFileClip(video_path, audio=True)

    if end_seconds is None or end_seconds > clip.duration:
        end_seconds = clip.duration

    clip = clip.subclipped(start_seconds, end_seconds)

    scale = OUTPUT_WIDTH / clip.w
    new_h = int(clip.h * scale)
    clip = clip.resized((OUTPUT_WIDTH, new_h))

    bg = ColorClip(size=(OUTPUT_WIDTH, OUTPUT_HEIGHT), color=(0, 0, 0))
    bg = bg.with_duration(clip.duration)

    y_offset = (OUTPUT_HEIGHT - new_h) // 2
    clip = clip.with_position((0, y_offset))

    layers = [bg, clip]

    if caption_words:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import numpy as np
        from moviepy import ImageClip
        try:
            caption_font = ImageFont.truetype("Helvetica", 48)
        except (OSError, IOError):
            try:
                caption_font = ImageFont.truetype("Arial", 48)
            except (OSError, IOError):
                caption_font = ImageFont.load_default()
        trimmed_words = [
            w for w in caption_words
            if w["end"] > start_seconds and w["start"] < end_seconds
        ]
        for w in trimmed_words:
            w_start = max(0.0, w["start"] - start_seconds)
            w_end = min(end_seconds - start_seconds, w["end"] - start_seconds)
            w_dur = w_end - w_start
            if w_dur <= 0:
                continue
            frame = PILImage.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            bbox = draw.textbbox((0, 0), w["word"], font=caption_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (OUTPUT_WIDTH - tw) // 2 - bbox[0]
            y = (OUTPUT_HEIGHT - th) // 2 - bbox[1]
            draw.text((x, y), w["word"], fill=(255, 255, 255, 255), font=caption_font)
            arr = np.array(frame)
            mask = ImageClip(arr[:, :, 3] / 255.0, is_mask=True)
            txt = ImageClip(arr[:, :, :3]).with_mask(mask)
            txt = txt.with_duration(w_dur).with_start(w_start)
            layers.append(txt)
        print(f"  Captions: {len(trimmed_words)} words")

    composite = CompositeVideoClip(layers, size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
    composite = composite.with_duration(clip.duration)
    composite.audio = clip.audio

    if composite.audio is None:
        print("WARNING: Intro clip has no audio track.")
    else:
        print(f"  Intro audio: {composite.audio.duration:.1f}s")

    return composite


def _trim_audio(audio_path, start=None, end=None):
    """Trim audio to the specified range using ffmpeg. Returns path to temp file."""
    import tempfile

    if start is None and end is None:
        return str(audio_path)

    ffmpeg = _get_ffmpeg_path()
    temp_path = str(Path(tempfile.gettempdir()) / "videosplice_trimmed_audio.wav")
    s = start if start is not None else 0.0

    cmd = [ffmpeg, "-hide_banner", "-y", "-ss", f"{s:.1f}", "-i", str(audio_path)]
    if end is not None:
        cmd += ["-t", f"{end - s:.1f}"]
    cmd.append(temp_path)

    print(f"Trimming audio: {s:.1f}s → {end:.1f}s..." if end else f"Trimming audio: {s:.1f}s → end...")
    subprocess.run(cmd, capture_output=True, check=True)
    return temp_path


# ===========================================================================
# 9. ARGUMENT PARSING
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
        "--audio-start",
        type=float,
        default=None,
        help="Start time in seconds for audio trimming (default: beginning of file)",
    )

    parser.add_argument(
        "--audio-end",
        type=float,
        default=None,
        help="End time in seconds for audio trimming (default: end of file)",
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

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override auto-detected BPM with a known value (sequence mode only).",
    )

    parser.add_argument(
        "--intro-url",
        help="YouTube URL for an intro clip prepended to the output (sequence mode only).",
    )

    parser.add_argument(
        "--intro-start",
        default="0:00",
        help="Start timestamp for the intro clip, e.g. '0:30' or '1:15' (default: 0:00).",
    )

    parser.add_argument(
        "--intro-end",
        help="End timestamp for the intro clip, e.g. '0:45' or '1:30'. Required with --intro-url.",
    )

    parser.add_argument(
        "--intro-overlap",
        action="store_true",
        help="Overlap the intro clip with the beginning of the main video, ducking audio.",
    )

    parser.add_argument(
        "--duck-db",
        type=float,
        default=-8.0,
        help="Amount of dB to duck the main audio during intro overlap (default: -8).",
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

    # Intro clip validation
    if arguments.intro_url:
        if not arguments.sequence_mode:
            parser.error("--intro-url requires --sequence-mode")
        try:
            _parse_timestamp(arguments.intro_start)
        except ValueError:
            parser.error(f"Invalid --intro-start timestamp: {arguments.intro_start}")
        if arguments.intro_end:
            try:
                _parse_timestamp(arguments.intro_end)
            except ValueError:
                parser.error(f"Invalid --intro-end timestamp: {arguments.intro_end}")

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
        input_folder = download_pinterest_board(
            arguments.pinterest,
            progress_callback=progress_callback,
            force_refresh=getattr(arguments, "force_refresh", False),
        )
    else:
        input_folder = arguments.input

    # Build output filenames from the audio file name
    audio_stem = Path(arguments.audio).stem
    output_dir = arguments.output_dir

    pipeline_start = time.time()

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

    # Trim audio if start/end specified
    audio_start = getattr(arguments, "audio_start", None)
    audio_end = getattr(arguments, "audio_end", None)
    audio_path = _trim_audio(arguments.audio, audio_start, audio_end)

    PREP_PCT = 5
    _progress(PREP_PCT if arguments.pinterest else 0, 100)

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

    _progress(PREP_PCT, 100)

    # Load audio duration once
    audio_for_duration_check = AudioFileClip(str(audio_path))
    target_total_duration = audio_for_duration_check.duration
    audio_for_duration_check.close()

    print(f"\nAudio duration: {target_total_duration:.1f}s — this will be the output video length.")

    # Analyze audio once if sequence mode is enabled
    transient_map = None
    if arguments.sequence_mode:
        user_bpm = getattr(arguments, "bpm", None)
        transient_map, _ = detect_all_transients(audio_path, kick_only=True, user_bpm=user_bpm)

    # Download and prepare intro clip if requested
    intro_clip = None
    intro_url = getattr(arguments, "intro_url", None)
    if intro_url:
        pre_downloaded = getattr(arguments, "intro_video_path", None)
        if pre_downloaded and os.path.isfile(pre_downloaded):
            intro_video_path = pre_downloaded
            print(f"  Using pre-downloaded intro video: {intro_video_path}")
        else:
            intro_video_path = download_intro_video(intro_url)
        start_sec = _parse_timestamp(getattr(arguments, "intro_start", "0:00"))
        intro_end = getattr(arguments, "intro_end", None)
        end_sec = _parse_timestamp(intro_end) if intro_end else None
        print(f"  Intro trim: {start_sec:.1f}s - {end_sec:.1f}s" if end_sec else f"  Intro trim: {start_sec:.1f}s - end")
        caption_words = getattr(arguments, "intro_caption_words", [])
        intro_clip = prepare_intro_clip(intro_video_path, start_sec, end_sec,
                                        caption_words=caption_words or None)
        print(f"  Intro clip ready: {intro_clip.duration:.1f}s")

        if getattr(arguments, "intro_overlap", False):
            if intro_clip.duration >= target_total_duration:
                print("ERROR: With --intro-overlap, the audio clip must be longer "
                      f"than the intro clip ({intro_clip.duration:.1f}s >= {target_total_duration:.1f}s).")
                sys.exit(1)
            print(f"  Overlap mode: audio will be ducked {arguments.duck_db} dB for first {intro_clip.duration:.1f}s")
            if transient_map:
                original_len = len(transient_map)
                transient_map = [(t, typ) for t, typ in transient_map if t >= intro_clip.duration]
                skipped = original_len - len(transient_map)
                if skipped:
                    print(f"  Skipped {skipped} beats during intro overlap ({intro_clip.duration:.1f}s)")

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
            intro_offset = 0.0
            if intro_clip and getattr(arguments, "intro_overlap", False):
                intro_offset = intro_clip.duration
            clip_sequence = assemble_sequence_mode(
                image_paths=image_paths,
                video_paths=video_paths,
                transient_map=transient_map,
                audio_duration=target_total_duration,
                intro_offset=intro_offset,
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

        video_idx = video_number - starting_number
        render_pct_per_video = (100 - PREP_PCT) / arguments.count
        base_pct = PREP_PCT + video_idx * render_pct_per_video

        def _render_cb(frac, _):
            pct = base_pct + frac * render_pct_per_video
            _progress(int(pct), 100)

        render_final_video(
            clip_sequence=clip_sequence,
            audio_file_path=audio_path,
            output_file_path=output_path,
            intro_clip=intro_clip,
            intro_overlap=getattr(arguments, "intro_overlap", False),
            duck_db=getattr(arguments, "duck_db", -8.0),
            render_progress_callback=_render_cb,
        )

        _progress(int(base_pct + render_pct_per_video), 100)

    # Clean up downloaded intro video (only if pipeline downloaded it, not pre-downloaded by UI)
    if intro_clip:
        intro_clip.close()
        pre_downloaded = getattr(arguments, "intro_video_path", None)
        if not pre_downloaded:
            cache_dir = Path(tempfile.gettempdir()) / "videosplice_intro"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
                print("Cleaned up downloaded intro video.")

    for seg_path in _SEGMENT_CACHE.values():
        try:
            os.remove(seg_path)
        except OSError:
            pass
    _SEGMENT_CACHE.clear()

    total_elapsed = time.time() - pipeline_start
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"\nAll done! Generated {arguments.count} video(s) in {output_dir}/")
    print(f"Total time: {int(minutes)}m {seconds:.1f}s")


def main():
    """CLI entry point — parse arguments and run the pipeline."""
    arguments = parse_command_line_arguments()
    run_pipeline(arguments)


# Standard Python idiom: only run main() when this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
