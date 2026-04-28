# Video Splice

Takes a folder of images and videos (or a Pinterest board), splices them together randomly, and overlays an audio track. Output is a vertical video (1080x1920) whose length matches the audio.

### Examples

Basic clip mode:

https://github.com/user-attachments/assets/394c5598-4da5-45c5-bb67-94ebdab5dc8c

Sequence mode (kick-synced transitions):

https://github.com/user-attachments/assets/bdc45554-cc09-47b5-aea7-17e2ac8f5168

## Setup

Create a conda environment with Python 3.11 and install dependencies. Using pip alone can cause build failures with `numba`/`llvmlite` and numpy version conflicts.

```bash
conda create -n videosplice python=3.11 numpy=1.26 numba -y
conda activate videosplice
pip install moviepy librosa gallery-dl "scenedetect[opencv]" "numpy==2.2.*"
```

## Usage

There are two ways to run Video Splice: a GUI or the CLI.

### GUI

```bash
python ui.py
```

The UI has fields for all options. The log panel shows the equivalent CLI command for every run so you can reproduce results in the terminal.

### CLI

#### Media input

Provide media from a local folder or a Pinterest board URL:

```bash
python video_splice.py -i ./media -a track.wav
python video_splice.py -p "https://www.pinterest.com/user/board-name/" -a track.wav
```

Pinterest boards are downloaded once via `gallery-dl` and cached locally. Board name matching is fuzzy — dashes, underscores, special characters, and extra whitespace are ignored.

#### Output

Files are named after the audio file: `track_1.mp4`, `track_2.mp4`, etc. Existing files are never overwritten — numbering continues from the next available number.

```bash
# Output to a specific directory
python video_splice.py -i ./media -a track.wav -o ./outputs

# Generate 5 videos at once (Pinterest is only downloaded once)
python video_splice.py -p "https://..." -a track.wav -o ./outputs --count 5
```

#### Sequence mode

```bash
python video_splice.py -i ./media -a track.wav --sequence-mode
```

Sequence mode detects kick drum hits in the audio and uses them as transition points for multi-phase visual sequences (picture-in-video, triple image, video halved, etc.). Each sequence type consumes a set number of kicks before the next sequence begins.

**Sequence mode only supports four-on-the-floor audio** where the kick hits on every beat. Irregular or syncopated kick patterns will produce inconsistent cuts.

When `--sequence-mode` is enabled it overrides `--clip-length`, `--image-length`, `--images-only`, and `--videos-only` — those flags cannot be used with sequence mode.

Videos are analyzed with scene detection so that multi-shot clips are split into individual shots. Images and videos smaller than 640,000 total pixels are excluded.

#### Landscape mode

```bash
python video_splice.py -i ./media -a track.wav --landscape
python video_splice.py -i ./media -a track.wav --sequence-mode --landscape
```

`--landscape` outputs 1920x1080 instead of the default 1080x1920. It can be combined with any other mode.

**Your input media must contain a large number of landscape-oriented images and videos.** In landscape mode, the tool strongly prefers media where the width is greater than the height. Portrait media is used as a fallback but will be cropped heavily, so having too few landscape sources will produce repetitive or poorly framed results.

In sequence mode, landscape orientation also affects how sequences are composed — overlays and splits use the horizontal axis instead of the vertical.

#### Clip timing (non-sequence mode)

Without `--sequence-mode`, clips and images use fixed durations:

```bash
python video_splice.py -i ./media -a track.wav --clip-length 3 --image-length 2
```

#### Media filters (non-sequence mode)

```bash
python video_splice.py -i ./media -a track.wav --images-only
python video_splice.py -i ./media -a track.wav --videos-only
```

#### Full examples

```bash
# Sequence mode from a Pinterest board, 3 videos
python video_splice.py \
  -p "https://www.pinterest.com/user/my-board/" \
  -a ./music.wav \
  -o ./outputs \
  --count 3 \
  --sequence-mode

# Landscape sequence mode from a local folder
python video_splice.py \
  -i ./media \
  -a ./music.wav \
  -o ./outputs \
  --sequence-mode \
  --landscape

# Fixed timing, images only
python video_splice.py \
  -i ./media \
  -a ./music.wav \
  -o ./outputs \
  --clip-length 3 \
  --image-length 2 \
  --images-only
```

## CLI reference

| Flag | Description | Default |
|---|---|---|
| `-i` / `--input` | Path to folder of images and videos | — |
| `-p` / `--pinterest` | Pinterest board URL (alternative to `-i`) | — |
| `-a` / `--audio` | Path to the audio file (.mp3 or .wav) | required |
| `-o` / `--output-dir` | Directory for output files | `~/Desktop` |
| `--count` | Number of videos to generate | `1` |
| `--sequence-mode` | Kick-based sequence editing (overrides clip/image timing) | off |
| `--landscape` | Output 1920x1080 instead of 1080x1920 | off |
| `--clip-length` | Seconds per video clip (non-sequence mode) | `1.0` |
| `--image-length` | Seconds per image (non-sequence mode) | `0.1` |
| `--images-only` | Only use images (non-sequence mode) | off |
| `--videos-only` | Only use videos (non-sequence mode) | off |

