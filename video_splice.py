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
# Constants — the target resolution for the output video (vertical / portrait)
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

    # Iterate over every file in the folder (non-recursive, top-level only)
    for entry in Path(input_folder_path).iterdir():
        if not entry.is_file():
            continue  # skip subdirectories

        # Lowercase the extension so ".JPG" and ".jpg" are treated the same
        file_extension = entry.suffix.lower()

        if file_extension in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(entry)
        elif file_extension in SUPPORTED_VIDEO_EXTENSIONS:
            video_paths.append(entry)
        # Files with other extensions are silently ignored

    total_media_count = len(image_paths) + len(video_paths)
    if total_media_count == 0:
        print(f"ERROR: No supported media files found in '{input_folder_path}'.")
        print(f"  Supported images: {SUPPORTED_IMAGE_EXTENSIONS}")
        print(f"  Supported videos: {SUPPORTED_VIDEO_EXTENSIONS}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) and {len(video_paths)} video(s).")
    return image_paths, video_paths


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
# 3. BEAT DETECTION
#    Analyze audio to find bass kick transients for beat-synced transitions.
# ===========================================================================

def detect_bass_kicks(audio_file_path, min_gap_seconds=0.3, transition_delay=0.03):
    """
    Analyze an audio file and return a list of beat transition points.

    Isolates the kick drum frequency range (30-120 Hz), computes the onset
    strength, and picks only peaks that exceed a relative threshold. This
    avoids triggering on quiet bass notes or bleed from other instruments.

    Returns a list of beat timestamps (in seconds) that serve as clip
    boundaries, starting with 0.0 and ending with the audio duration.
    """
    print("\nAnalyzing audio for bass kicks...")

    audio_samples, sample_rate = librosa.load(str(audio_file_path), sr=None, mono=True)
    audio_duration = librosa.get_duration(y=audio_samples, sr=sample_rate)

    # Isolate kick drum frequencies (30-120 Hz) via bandpass in the STFT domain
    stft = librosa.stft(audio_samples)
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    bass_mask = (frequencies >= 30) & (frequencies <= 120)
    bass_stft = np.zeros_like(stft)
    bass_stft[bass_mask, :] = stft[bass_mask, :]
    bass_signal = librosa.istft(bass_stft)

    # Compute onset strength envelope from the bass signal
    onset_envelope = librosa.onset.onset_strength(y=bass_signal, sr=sample_rate)

    # Use a relative threshold: only peaks above 60% of the envelope's
    # standard deviation are considered kicks. This filters out low-energy
    # bass rumble and ghost notes.
    threshold = np.mean(onset_envelope) + 0.6 * np.std(onset_envelope)

    onset_frames = librosa.onset.onset_detect(
        y=bass_signal, sr=sample_rate,
        onset_envelope=onset_envelope,
        backtrack=False,
        delta=threshold,
        wait=int(min_gap_seconds * sample_rate / 512),
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    # Add a small delay so the transition lands just after the kick
    onset_times = onset_times + transition_delay

    # Enforce minimum gap between transitions
    filtered_times = [0.0]
    for time in onset_times:
        if time - filtered_times[-1] >= min_gap_seconds:
            filtered_times.append(min(time, audio_duration))

    if filtered_times[-1] < audio_duration:
        filtered_times.append(audio_duration)

    print(f"  Found {len(filtered_times) - 1} beat segments")
    if len(filtered_times) > 2:
        durations = [filtered_times[i+1] - filtered_times[i] for i in range(len(filtered_times) - 1)]
        print(f"  Shortest: {min(durations):.2f}s, Longest: {max(durations):.2f}s, Average: {np.mean(durations):.2f}s")

    return filtered_times


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


def build_video_clip(video_path, max_clip_duration_seconds):
    """
    Create a moviepy clip from a video file, trimmed to the requested length.

    If the source video is shorter than `max_clip_duration_seconds`, the clip
    uses the full video length (we don't loop it). If it's longer, we take
    a random sub-segment of the requested duration.
    """
    # Load the full source video (without audio — we'll add our own later)
    source_video = VideoFileClip(str(video_path), audio=False)

    source_duration = source_video.duration

    if source_duration <= max_clip_duration_seconds:
        # Source is shorter than requested — use the entire clip as-is
        trimmed_clip = source_video
    else:
        # Source is longer — pick a random start point and extract a segment
        latest_possible_start = source_duration - max_clip_duration_seconds
        random_start_time = random.uniform(0, latest_possible_start)
        random_end_time = random_start_time + max_clip_duration_seconds

        trimmed_clip = source_video.subclipped(random_start_time, random_end_time)

    # Scale and crop to fill the output frame
    fitted_clip = resize_clip_to_fill_frame(trimmed_clip)

    return fitted_clip


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
    beat_times=None,
):
    """
    Build a list of moviepy clips that, when concatenated, fill the target duration.

    If beat_times is provided (from beat-sync mode), it's a list of timestamp
    boundaries. Images span 1 beat gap, videos span 2 beat gaps so they play longer.

    Returns a list of moviepy clips ready for concatenation.
    """
    if not image_paths and not video_paths:
        print("ERROR: No media files provided. Cannot assemble sequence.")
        sys.exit(1)

    IMAGE_GROUP_SIZE = 10

    clip_sequence = []
    accumulated_duration = 0.0
    clip_index = 0
    unused_images = list(image_paths)
    unused_videos = list(video_paths)
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
                unused_videos = list(video_paths)
                random.shuffle(unused_videos)
            print("  (all media used once — reshuffling)")

    if beat_times:
        beat_index = 0
        last_beat = len(beat_times) - 1

        while beat_index < last_beat:
            _refill_if_both_empty()

            eligible_types = []
            if unused_videos:
                eligible_types.append("video")
            if unused_images:
                eligible_types.append("image")

            chosen_type = random.choice(eligible_types)

            if chosen_type == "image":
                for _ in range(IMAGE_GROUP_SIZE):
                    if beat_index >= last_beat:
                        break
                    _refill_if_both_empty()
                    if not unused_images:
                        break

                    image_path = unused_images.pop()
                    duration = beat_times[beat_index + 1] - beat_times[beat_index]
                    beat_index += 1

                    clip_index += 1
                    print(f"  Clip {clip_index}: image — {image_path.name} ({duration:.2f}s) [beat {beat_index}]")

                    new_clip = build_image_clip(image_path, duration)
                    clip_sequence.append(new_clip)
                    accumulated_duration += new_clip.duration

                if beat_index < last_beat and unused_videos:
                    video_path = unused_videos.pop()
                    end_beat = min(beat_index + 3, last_beat)
                    duration = beat_times[end_beat] - beat_times[beat_index]
                    beat_index = end_beat

                    clip_index += 1
                    print(f"  Clip {clip_index}: video — {video_path.name} ({duration:.2f}s) [beats {beat_index - 1}-{beat_index}]")

                    new_clip = build_video_clip(video_path, duration)
                    clip_sequence.append(new_clip)
                    accumulated_duration += new_clip.duration
            else:
                video_path = unused_videos.pop()
                end_beat = min(beat_index + 3, last_beat)
                duration = beat_times[end_beat] - beat_times[beat_index]
                beat_index = end_beat

                clip_index += 1
                print(f"  Clip {clip_index}: video — {video_path.name} ({duration:.2f}s) [beats {beat_index - 1}-{beat_index}]")

                new_clip = build_video_clip(video_path, duration)
                clip_sequence.append(new_clip)
                accumulated_duration += new_clip.duration
    else:
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
                    video_path = unused_videos.pop()
                    actual_duration = min(clip_duration_seconds, remaining_duration)

                    clip_index += 1
                    print(f"  Clip {clip_index}: video — {video_path.name} ({actual_duration:.1f}s)")

                    new_clip = build_video_clip(video_path, actual_duration)
                    clip_sequence.append(new_clip)
                    accumulated_duration += new_clip.duration
            else:
                video_path = unused_videos.pop()
                actual_duration = min(clip_duration_seconds, remaining_duration)

                clip_index += 1
                print(f"  Clip {clip_index}: video — {video_path.name} ({actual_duration:.1f}s)")

                new_clip = build_video_clip(video_path, actual_duration)
                clip_sequence.append(new_clip)
                accumulated_duration += new_clip.duration

    print(f"\nTotal assembled duration: {accumulated_duration:.1f}s")
    return clip_sequence


# ===========================================================================
# 5. FINAL VIDEO RENDER
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

GALLERY_DL_ROOT = Path("/Users/wesleyolmsted/video-splice/gallery-dl/pinterest")


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
    Reduce a board name to a comparable form by lowercasing and stripping
    all punctuation/whitespace. This handles mismatches between the URL
    slug and the folder name gallery-dl creates (e.g. hyphens vs pipes,
    spaces, capitalization differences).
    """
    import re
    return re.sub(r'[\s\-_|]+', '', name).lower()


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

    return None


def download_pinterest_board(pinterest_url):
    """
    Download all images/videos from a Pinterest board using gallery-dl.
    Returns the path to the folder where the media was saved.
    """
    username, board_name = parse_pinterest_url(pinterest_url)

    print(f"\nDownloading Pinterest board: {pinterest_url}")
    print(f"  Username: {username}")
    print(f"  Board name: {board_name}")
    print("  Running gallery-dl (this may take a while)...")

    # Count files before download so we can report what's new
    board_folder = find_board_folder(username, board_name)
    files_before = set(board_folder.iterdir()) if board_folder else set()

    result = subprocess.run(
        ["gallery-dl", pinterest_url],
        capture_output=True,
        text=True,
    )

    # Re-resolve board folder after download (may have just been created)
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
        default=".",
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
        "--beat-sync",
        action="store_true",
        help="Sync clip transitions to bass kicks in the audio. Overrides --clip-length and --image-length.",
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

    # Clip and image lengths must be positive
    if arguments.clip_length <= 0:
        parser.error("--clip-length must be a positive number")
    if arguments.image_length <= 0:
        parser.error("--image-length must be a positive number")

    return arguments


# ===========================================================================
# 7. MAIN — tie everything together
# ===========================================================================

def run_pipeline(arguments):
    """
    Run the full video splice pipeline with the given arguments object.
    Can be called from CLI parsing or from the GUI.
    """
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
    if arguments.beat_sync:
        print(f"  Beat sync    : on (overrides clip/image length)")
    else:
        print(f"  Clip length  : {arguments.clip_length}s")
        print(f"  Image length : {arguments.image_length}s")
    media_filter = "images only" if arguments.images_only else "videos only" if arguments.videos_only else "all"
    print(f"  Media filter : {media_filter}")
    print("=" * 60)

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

    # Load audio duration once
    audio_for_duration_check = AudioFileClip(str(arguments.audio))
    target_total_duration = audio_for_duration_check.duration
    audio_for_duration_check.close()

    print(f"\nAudio duration: {target_total_duration:.1f}s — this will be the output video length.")

    # Detect beats once if beat-sync is enabled
    beat_times = None
    if arguments.beat_sync:
        beat_times = detect_bass_kicks(arguments.audio)

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

        clip_sequence = assemble_clip_sequence(
            image_paths=image_paths,
            video_paths=video_paths,
            target_total_duration_seconds=target_total_duration,
            clip_duration_seconds=arguments.clip_length,
            image_duration_seconds=arguments.image_length,
            beat_times=beat_times,
        )

        render_final_video(
            clip_sequence=clip_sequence,
            audio_file_path=arguments.audio,
            output_file_path=output_path,
        )

    print(f"\nAll done! Generated {arguments.count} video(s) in {output_dir}/")


def main():
    """CLI entry point — parse arguments and run the pipeline."""
    arguments = parse_command_line_arguments()
    run_pipeline(arguments)


# Standard Python idiom: only run main() when this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
