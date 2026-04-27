# Video Splice

Takes a folder of images and videos (or a Pinterest board), splices them together randomly, and overlays an audio track. Output is a vertical video (1080x1920) whose length matches the audio.

## Setup

```bash
pip install moviepy librosa gallery-dl
```

## GUI

```bash
python ui.py
```

The UI has fields for all options. The log panel shows the equivalent CLI command for every run so you can reproduce results in the terminal.

## CLI

### Basic usage

From a local folder:
```bash
python video_splice.py -i ./media -a track.wav
```

From a Pinterest board:
```bash
python video_splice.py -p "https://www.pinterest.com/user/board-name/" -a track.wav
```

### Output

Files are named after the audio file: `track_1.mp4`, `track_2.mp4`, etc. Existing files are never overwritten -- numbering continues from the next available number.

```bash
# Output to a specific directory
python video_splice.py -i ./media -a track.wav -o ./outputs

# Generate 5 videos at once (Pinterest is only downloaded once)
python video_splice.py -p "https://..." -a track.wav -o ./outputs --count 5
```

### Clip timing

Fixed durations (default):
```bash
python video_splice.py -i ./media -a track.wav --clip-length 3 --image-length 2
```

Beat-synced transitions (overrides clip/image length):
```bash
python video_splice.py -i ./media -a track.wav --beat-sync
```

`--beat-sync` analyzes bass frequencies in the audio and cuts on kick drum hits. Images transition every beat; videos span 3 beats.

### Media filters

```bash
python video_splice.py -i ./media -a track.wav --images-only
python video_splice.py -i ./media -a track.wav --videos-only
```

### Full example

```bash
python video_splice.py \
  -p "https://www.pinterest.com/user/my-board/" \
  -a ./music.wav \
  -o ./outputs \
  --count 3 \
  --beat-sync \
  --images-only
```
