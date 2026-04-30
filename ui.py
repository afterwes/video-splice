"""
ui.py — Tkinter GUI for Video Splice.

Launch with: python ui.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import bundle_runtime
bundle_runtime.setup()
import threading
import tkinter as tk
from argparse import Namespace
from tkinter import filedialog, messagebox, scrolledtext, font, ttk

from PIL import Image, ImageTk
from video_splice import run_pipeline, _get_ffmpeg_path

BG = "#F0EDE8"
FG = "#1A1A1A"
MUTED = "#999999"
DIVIDER = "#D4D0CA"
ENTRY_BG = "#FFFFFF"
BTN_BG = "#1A1A1A"
BTN_FG = "#F0EDE8"
BTN_DISABLED_FG = "#666666"


ACCENT = "#4A90D9"
TAB_INACTIVE_BG = "#E4E1DC"

def _format_time(seconds):
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"

def _parse_time(text):
    text = text.strip()
    if ":" in text:
        parts = text.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(text)


class AudioTrimmer(tk.Frame):
    """Range slider with start/end entries and audio preview."""

    HANDLE_R = 8
    PAD = 14
    WAVE_H = 50
    WAVE_COLOR = "#B8B3AC"
    WAVE_ACTIVE = "#7A7570"
    OVERLAP_COLOR = "#5B9BD5"
    OVERLAP_BG = "#D6E8F7"

    def __init__(self, parent, fonts, get_overlap_info=None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self.duration = 0.0
        self.start_val = 0.0
        self.end_val = 0.0
        self._dragging = None
        self._audio_path = None
        self._preview_proc = None
        self._waveform = None
        self._loading = False
        self._loading_dots = 0
        self._fonts = fonts
        self._playback_start_time = None
        self._playback_cursor_active = False
        self._overlap_duration = 0.0
        self._get_overlap_info = get_overlap_info

        self.slider = tk.Canvas(self, bg=BG, height=self.WAVE_H + 16, highlightthickness=0, cursor="hand2")
        self.slider.bind("<Button-1>", self._on_press)
        self.slider.bind("<B1-Motion>", self._on_drag)
        self.slider.bind("<ButtonRelease-1>", self._on_release)
        self.slider.bind("<Configure>", lambda e: self._draw())

        self._controls = tk.Frame(self, bg=BG)

        tk.Label(self._controls, text="Start", font=fonts["small"], bg=BG, fg=MUTED).pack(side="left")
        self.start_entry = tk.Entry(
            self._controls, font=fonts["body"], bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=7,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER, insertbackground=FG,
        )
        self.start_entry.pack(side="left", padx=(4, 16), ipady=3)
        self.start_entry.bind("<Return>", self._on_start_entry)
        self.start_entry.bind("<FocusOut>", self._on_start_entry)

        tk.Label(self._controls, text="End", font=fonts["small"], bg=BG, fg=MUTED).pack(side="left")
        self.end_entry = tk.Entry(
            self._controls, font=fonts["body"], bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=7,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER, insertbackground=FG,
        )
        self.end_entry.pack(side="left", padx=(4, 16), ipady=3)
        self.end_entry.bind("<Return>", self._on_end_entry)
        self.end_entry.bind("<FocusOut>", self._on_end_entry)

        self.duration_label = tk.Label(
            self._controls, text="", font=fonts["small"], bg=BG, fg=MUTED,
        )
        self.duration_label.pack(side="left", padx=(0, 16))

        self.preview_label = tk.Label(
            self._controls, text="Preview", font=fonts["small"],
            bg=BG, fg=ACCENT, cursor="hand2",
        )
        self.preview_label.pack(side="right")
        self.preview_label.bind("<Button-1>", lambda e: self._toggle_preview())

    def set_overlap_duration(self, duration):
        self._overlap_duration = max(0.0, duration)
        self._draw()

    def set_audio(self, path, duration):
        self._audio_path = path
        self.duration = duration
        self.start_val = 0.0
        self.end_val = duration
        self._waveform = None
        self._loading = True
        self._loading_dots = 0
        self._show()
        self._sync_entries()
        self._draw()
        self._animate_loading()
        threading.Thread(target=self._compute_waveform, args=(path,), daemon=True).start()

    def _show(self):
        self.slider.pack(fill="x", pady=(6, 2))
        self._controls.pack(fill="x", pady=(0, 2))

    def _animate_loading(self):
        if not self._loading:
            return
        self._loading_dots = (self._loading_dots + 1) % 4
        self._draw()
        self.after(400, self._animate_loading)

    def _compute_waveform(self, path):
        try:
            import librosa
            import numpy as np
            y, sr = librosa.load(path, sr=8000, mono=True)
            num_bins = 800
            bin_size = max(1, len(y) // num_bins)
            envelope = []
            for i in range(0, len(y), bin_size):
                chunk = y[i:i + bin_size]
                envelope.append(float(np.sqrt(np.mean(chunk ** 2))))
            peak = max(envelope) if envelope else 1.0
            if peak > 0:
                envelope = [v / peak for v in envelope]
            self._waveform = envelope
        except Exception:
            pass
        self._loading = False
        self.slider.after(0, self._draw)

    def _val_to_x(self, val):
        w = self.slider.winfo_width()
        usable = w - 2 * self.PAD
        if self.duration <= 0 or usable <= 0:
            return self.PAD
        return self.PAD + (val / self.duration) * usable

    def _x_to_val(self, x):
        w = self.slider.winfo_width()
        usable = w - 2 * self.PAD
        if usable <= 0:
            return 0.0
        val = ((x - self.PAD) / usable) * self.duration
        return round(max(0.0, min(self.duration, val)), 1)

    def _draw(self):
        c = self.slider
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 30:
            return
        mid_y = h // 2
        usable = w - 2 * self.PAD

        if self._loading and not self._waveform:
            dots = "." * self._loading_dots
            c.create_text(w // 2, mid_y, text=f"Loading waveform{dots}",
                          font=self._fonts["small"], fill=MUTED)
            return

        x0 = self._val_to_x(self.start_val)
        x1 = self._val_to_x(self.end_val)

        overlap_end_val = self.start_val + self._overlap_duration
        ox1 = self._val_to_x(overlap_end_val) if self._overlap_duration > 0 else x0
        has_overlap = self._overlap_duration > 0 and overlap_end_val <= self.end_val

        if has_overlap:
            half_h = self.WAVE_H // 2
            c.create_rectangle(
                x0, mid_y - half_h - 2, ox1, mid_y + half_h + 2,
                fill=self.OVERLAP_BG, outline="",
            )

        if self._waveform and usable > 0:
            num_bars = len(self._waveform)
            bar_w = max(1, usable / num_bars)
            half_h = self.WAVE_H // 2
            for i, amp in enumerate(self._waveform):
                bx = self.PAD + (i / num_bars) * usable
                bar_h = max(1, int(amp * half_h))
                bar_center = bx + bar_w / 2
                in_range = x0 <= bar_center <= x1
                if has_overlap and x0 <= bar_center <= ox1:
                    color = self.OVERLAP_COLOR
                elif in_range:
                    color = self.WAVE_ACTIVE
                else:
                    color = self.WAVE_COLOR
                c.create_rectangle(
                    bx, mid_y - bar_h, bx + bar_w, mid_y + bar_h,
                    fill=color, outline="",
                )
        else:
            c.create_rectangle(self.PAD, mid_y - 1, w - self.PAD, mid_y + 1,
                               fill=DIVIDER, outline="")

        c.create_line(x0, mid_y, x1, mid_y, fill=FG, width=2)

        r = self.HANDLE_R
        c.create_oval(x0 - r, mid_y - r, x0 + r, mid_y + r, fill=FG, outline="")
        c.create_oval(x1 - r, mid_y - r, x1 + r, mid_y + r, fill=FG, outline="")

        if self._playback_cursor_active and self._playback_start_time is not None:
            import time as _time
            elapsed = _time.time() - self._playback_start_time
            cursor_val = self.start_val + elapsed
            if cursor_val <= self.end_val:
                cx = self._val_to_x(cursor_val)
                half_h = self.WAVE_H // 2 + 4
                c.create_line(cx, mid_y - half_h, cx, mid_y + half_h,
                              fill=ACCENT, width=2)

    def _on_press(self, event):
        x0 = self._val_to_x(self.start_val)
        x1 = self._val_to_x(self.end_val)
        d_start = abs(event.x - x0)
        d_end = abs(event.x - x1)
        if d_start <= d_end and d_start < 20:
            self._dragging = "start"
        elif d_end < 20:
            self._dragging = "end"

    def _on_drag(self, event):
        if not self._dragging:
            return
        val = self._x_to_val(event.x)
        if self._dragging == "start":
            self.start_val = min(val, self.end_val - 0.1)
        else:
            self.end_val = max(val, self.start_val + 0.1)
        self._sync_entries()
        self._draw()

    def _on_release(self, event):
        self._dragging = None

    def _on_start_entry(self, event=None):
        try:
            val = round(_parse_time(self.start_entry.get()), 1)
            self.start_val = max(0.0, min(val, self.end_val - 0.1))
        except (ValueError, IndexError):
            pass
        self._sync_entries()
        self._draw()

    def _on_end_entry(self, event=None):
        try:
            val = round(_parse_time(self.end_entry.get()), 1)
            self.end_val = max(self.start_val + 0.1, min(val, self.duration))
        except (ValueError, IndexError):
            pass
        self._sync_entries()
        self._draw()

    def _sync_entries(self):
        for entry, val in [(self.start_entry, self.start_val), (self.end_entry, self.end_val)]:
            entry.delete(0, tk.END)
            entry.insert(0, _format_time(val))
        clip_dur = self.end_val - self.start_val
        self.duration_label.configure(text=f"Duration: {_format_time(clip_dur)}")

    def _toggle_preview(self):
        if self._preview_proc and self._preview_proc.poll() is None:
            self._stop_preview()
            return
        self._play_preview()

    def _play_preview(self):
        if not self._audio_path:
            return
        self._stop_preview()
        ffmpeg = _get_ffmpeg_path()
        temp_path = os.path.join(tempfile.gettempdir(), "videosplice_preview.wav")
        start = self.start_val
        dur = self.end_val - self.start_val

        overlap_info = self._get_overlap_info() if self._get_overlap_info else None
        if overlap_info and self._overlap_duration > 0:
            intro_path, intro_start, intro_end, duck_db = overlap_info
            overlap_dur = min(self._overlap_duration, dur)
            duck_db = -abs(duck_db)
            intro_boost_db = 10.0
            intro_dur = intro_end - intro_start
            end_time = start + dur
            filter_parts = [
                f"[0:a]atrim={start:.3f}:{start + overlap_dur:.3f},asetpts=PTS-STARTPTS,volume={duck_db}dB[main_duck]",
                f"[0:a]atrim={start + overlap_dur:.3f}:{end_time:.3f},asetpts=PTS-STARTPTS[main_rest]",
                f"[1:a]atrim={intro_start:.3f}:{intro_end:.3f},asetpts=PTS-STARTPTS,volume={intro_boost_db}dB[intro]",
                f"[main_duck][intro]amix=inputs=2:duration=first:normalize=0[mixed]",
                f"[mixed][main_rest]concat=n=2:v=0:a=1[out]",
            ]
            filter_str = ";".join(filter_parts)
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-y",
                 "-i", self._audio_path,
                 "-i", intro_path,
                 "-filter_complex", filter_str, "-map", "[out]",
                 temp_path],
                capture_output=True,
            )
            if result.returncode != 0:
                subprocess.run(
                    [ffmpeg, "-hide_banner", "-y", "-ss", f"{start:.1f}",
                     "-i", self._audio_path, "-t", f"{dur:.1f}", temp_path],
                    capture_output=True,
                )
        else:
            subprocess.run(
                [ffmpeg, "-hide_banner", "-y", "-ss", f"{start:.1f}",
                 "-i", self._audio_path, "-t", f"{dur:.1f}", temp_path],
                capture_output=True,
            )

        self.preview_label.configure(text="Stop", fg="#CC3333")
        import time as _time
        self._playback_start_time = _time.time()
        self._playback_cursor_active = True
        self._tick_cursor()
        if sys.platform == "darwin":
            self._preview_proc = subprocess.Popen(["afplay", temp_path])
        elif sys.platform == "win32":
            self._preview_proc = subprocess.Popen(
                ["cmd", "/c", "start", "/min", "", temp_path],
                shell=False,
            )
        else:
            self._preview_proc = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-i", temp_path, "-f", "pulse", "-"],
                capture_output=True,
            )
        threading.Thread(target=self._wait_preview, daemon=True).start()

    def _tick_cursor(self):
        if not self._playback_cursor_active:
            return
        import time as _time
        elapsed = _time.time() - self._playback_start_time
        if self.start_val + elapsed >= self.end_val:
            self._stop_preview()
            return
        self._draw()
        self.after(30, self._tick_cursor)

    def _wait_preview(self):
        if self._preview_proc:
            self._preview_proc.wait()
        self._playback_cursor_active = False
        self._playback_start_time = None
        try:
            self.slider.after(0, self._draw)
            self.preview_label.after(0, lambda: self.preview_label.configure(text="Preview", fg=ACCENT))
        except tk.TclError:
            pass

    def _stop_preview(self):
        self._playback_cursor_active = False
        self._playback_start_time = None
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.terminate()
            try:
                self._preview_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._preview_proc.kill()
            self._preview_proc = None
        self.preview_label.configure(text="Preview", fg=ACCENT)
        self._draw()


class IntroClipPreview(tk.Frame):
    """Video preview with range slider for intro clip selection."""

    HANDLE_R = 8
    PAD = 14
    MAX_W = 750
    MAX_H = 420

    def __init__(self, parent, fonts, on_range_change=None, on_load_start=None,
                 on_load_complete=None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._fonts = fonts
        self._on_range_change = on_range_change
        self._on_load_start = on_load_start
        self._on_load_complete = on_load_complete
        self._video_clip = None
        self.downloaded_path = None
        self._display_photo = None
        self._preview_proc = None
        self._playback_active = False
        self._playback_start_time = None
        self._dragging = None
        self._display_w = 0
        self._display_h = 0
        self.duration = 0.0
        self.start_val = 0.0
        self.end_val = 0.0
        self._downloading = False
        self._downloading_dots = 0
        self._caption_words = []

        # URL row
        url_row = tk.Frame(self, bg=BG)
        url_row.pack(fill="x", pady=(2, 4))
        self.url_entry = tk.Entry(
            url_row, font=fonts["body"], bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm", state="disabled",
        )
        self.url_entry.pack(side="left", fill="x", expand=True, ipady=6)
        load_frame = tk.Frame(url_row, bg=DIVIDER, padx=1, pady=1)
        load_frame.pack(side="left", padx=(10, 0))
        self._load_label = tk.Label(
            load_frame, text="Load", font=fonts["small"],
            bg=BG, fg=FG, padx=10, pady=2, cursor="hand2",
        )
        self._load_label.pack()
        self._load_label.bind("<Button-1>", lambda e: self._start_load())
        load_frame.bind("<Button-1>", lambda e: self._start_load())

        caption_row = tk.Frame(self, bg=BG)
        caption_row.pack(fill="x")
        yt_link = tk.Label(caption_row, text="YouTube", font=fonts["small"],
                           bg=BG, fg=ACCENT, cursor="hand2")
        yt_link.pack(side="left")
        yt_link.bind("<Button-1>", lambda e: __import__("webbrowser").open("https://www.youtube.com/"))
        tk.Label(caption_row, text=" or ", font=fonts["small"],
                 bg=BG, fg=MUTED).pack(side="left")
        yarn_link = tk.Label(caption_row, text="Yarn", font=fonts["small"],
                             bg=BG, fg=ACCENT, cursor="hand2")
        yarn_link.pack(side="left")
        yarn_link.bind("<Button-1>", lambda e: __import__("webbrowser").open("https://yarn.co"))
        tk.Label(caption_row, text=" URL. Load to preview and select timestamps.",
                 font=fonts["small"], bg=BG, fg=MUTED).pack(side="left")

        self._status_var = tk.StringVar(value="")
        self._status_label = tk.Label(
            self, textvariable=self._status_var,
            font=fonts["small"], bg=BG, fg=MUTED, anchor="w",
        )

        self._video_label = tk.Label(self, bg="#000000")

        self._slider = tk.Canvas(self, bg=BG, height=36, highlightthickness=0, cursor="hand2")
        self._slider.bind("<Button-1>", self._on_press)
        self._slider.bind("<B1-Motion>", self._on_drag)
        self._slider.bind("<ButtonRelease-1>", self._on_release)
        self._slider.bind("<Configure>", lambda e: self._draw_slider())

        self._controls = tk.Frame(self, bg=BG)
        tk.Label(self._controls, text="Start", font=fonts["small"], bg=BG, fg=MUTED).pack(side="left")
        self.start_entry = tk.Entry(
            self._controls, font=fonts["body"], bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=7,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER, insertbackground=FG,
        )
        self.start_entry.pack(side="left", padx=(4, 16), ipady=3)
        self.start_entry.bind("<Return>", self._on_start_entry)
        self.start_entry.bind("<FocusOut>", self._on_start_entry)

        tk.Label(self._controls, text="End", font=fonts["small"], bg=BG, fg=MUTED).pack(side="left")
        self.end_entry = tk.Entry(
            self._controls, font=fonts["body"], bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=7,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER, insertbackground=FG,
        )
        self.end_entry.pack(side="left", padx=(4, 16), ipady=3)
        self.end_entry.bind("<Return>", self._on_end_entry)
        self.end_entry.bind("<FocusOut>", self._on_end_entry)

        self._duration_label = tk.Label(
            self._controls, text="", font=fonts["small"], bg=BG, fg=MUTED,
        )
        self._duration_label.pack(side="left", padx=(0, 16))

        self._preview_btn = tk.Label(
            self._controls, text="Preview", font=fonts["small"],
            bg=BG, fg=ACCENT, cursor="hand2",
        )
        self._preview_btn.pack(side="right")
        self._preview_btn.bind("<Button-1>", lambda e: self._toggle_preview())

    def enable(self):
        self.url_entry.configure(state="normal")

    def disable(self):
        self._stop_preview()
        self.url_entry.configure(state="disabled")

    # -- Download --

    def _start_load(self):
        url = self.url_entry.get().strip()
        if not url:
            return
        if self.url_entry.cget("state") == "disabled":
            return
        self._close_video()
        self.duration = 0.0
        self.start_val = 0.0
        self.end_val = 0.0
        self._caption_words = []
        if self._on_load_start:
            self._on_load_start()
        self._downloading = True
        self._downloading_dots = 0
        self._status_var.set("Downloading")
        self._status_label.pack(fill="x", pady=(4, 0))
        self._animate_downloading()
        threading.Thread(target=self._download, args=(url,), daemon=True).start()

    def _animate_downloading(self):
        if not self._downloading:
            return
        self._downloading_dots = (self._downloading_dots + 1) % 4
        self._status_var.set("Downloading" + "." * self._downloading_dots)
        self.after(400, self._animate_downloading)

    def _download(self, url):
        try:
            from video_splice import download_intro_video
            import io, contextlib
            writer = io.StringIO()
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                path = download_intro_video(url)
            self._downloading = False
            self.downloaded_path = path
            self.after(0, lambda: self._on_loaded(path))
        except SystemExit:
            self._downloading = False
            self.after(0, lambda: self._status_var.set("Download failed."))
        except Exception as e:
            self._downloading = False
            self.after(0, lambda: self._status_var.set(f"Error: {e}"))

    def _on_loaded(self, path):
        from moviepy import VideoFileClip
        try:
            self._video_clip = VideoFileClip(path, audio=False)
        except (OSError, IndexError):
            clean_path = path + ".clean.mp4"
            ffmpeg = _get_ffmpeg_path()
            subprocess.run(
                [ffmpeg, "-hide_banner", "-y", "-i", path,
                 "-map_chapters", "-1", "-c", "copy", clean_path],
                capture_output=True,
            )
            self.downloaded_path = clean_path
            self._video_clip = VideoFileClip(clean_path, audio=False)
        self.duration = self._video_clip.duration
        self.start_val = 0.0
        self.end_val = self.duration

        vid_w, vid_h = self._video_clip.size
        aspect = vid_h / vid_w
        self._display_w = min(self.MAX_W, vid_w)
        self._display_h = int(self._display_w * aspect)
        if self._display_h > self.MAX_H:
            self._display_h = self.MAX_H
            self._display_w = int(self._display_h / aspect)

        self._status_label.pack_forget()
        self._video_label.pack(fill="x", pady=(4, 2))
        self._slider.pack(fill="x", pady=(2, 2))
        self._controls.pack(fill="x", pady=(0, 4))
        self._sync_entries()
        self._show_frame_at(0.0)
        self._draw_slider()
        if self._on_load_complete:
            self._on_load_complete()

    def _close_video(self):
        self._stop_preview()
        if self._video_clip:
            self._video_clip.close()
            self._video_clip = None
        self._video_label.pack_forget()
        self._slider.pack_forget()
        self._controls.pack_forget()

    # -- Frame display --

    def _show_frame_at(self, t):
        if not self._video_clip:
            return
        t = max(0.0, min(t, self._video_clip.duration - 0.01))
        for attempt in range(2):
            try:
                frame = self._video_clip.get_frame(t)
                break
            except Exception:
                self._reopen_video()
                if not self._video_clip:
                    return
        else:
            return
        img = Image.fromarray(frame)
        img = img.resize((self._display_w, self._display_h), Image.LANCZOS)
        if self._caption_words:
            self._draw_caption_on_image(img, t)
        self._display_photo = ImageTk.PhotoImage(img)
        self._video_label.configure(image=self._display_photo)

    def _draw_caption_on_image(self, img, t):
        from PIL import ImageDraw, ImageFont
        word_text = None
        for w in self._caption_words:
            if w["start"] <= t < w["end"]:
                word_text = w["word"]
                break
        if not word_text:
            return
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Helvetica", 24)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("Arial", 24)
            except (OSError, IOError):
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), word_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (img.width - tw) // 2
        y = (img.height - th) // 2
        draw.text((x, y), word_text, fill="white", font=font)

    def _reopen_video(self):
        if not self.downloaded_path or not os.path.isfile(self.downloaded_path):
            return
        try:
            if self._video_clip:
                self._video_clip.close()
            from moviepy import VideoFileClip
            self._video_clip = VideoFileClip(self.downloaded_path, audio=False)
        except Exception:
            self._video_clip = None

    # -- Slider --

    def _val_to_x(self, val):
        w = self._slider.winfo_width()
        usable = w - 2 * self.PAD
        if self.duration <= 0 or usable <= 0:
            return self.PAD
        return self.PAD + (val / self.duration) * usable

    def _x_to_val(self, x):
        w = self._slider.winfo_width()
        usable = w - 2 * self.PAD
        if usable <= 0:
            return 0.0
        val = ((x - self.PAD) / usable) * self.duration
        return round(max(0.0, min(self.duration, val)), 1)

    def _draw_slider(self):
        c = self._slider
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 30 or self.duration <= 0:
            return
        mid_y = h // 2

        c.create_rectangle(self.PAD, mid_y - 2, w - self.PAD, mid_y + 2,
                           fill=DIVIDER, outline="")

        x0 = self._val_to_x(self.start_val)
        x1 = self._val_to_x(self.end_val)
        c.create_line(x0, mid_y, x1, mid_y, fill=FG, width=2)

        r = self.HANDLE_R
        c.create_oval(x0 - r, mid_y - r, x0 + r, mid_y + r, fill=FG, outline="")
        c.create_oval(x1 - r, mid_y - r, x1 + r, mid_y + r, fill=FG, outline="")

        if self._playback_active and self._playback_start_time is not None:
            import time as _time
            elapsed = _time.time() - self._playback_start_time
            cursor_val = self.start_val + elapsed
            if cursor_val <= self.end_val:
                cx = self._val_to_x(cursor_val)
                c.create_line(cx, mid_y - 14, cx, mid_y + 14, fill=ACCENT, width=2)

    def _on_press(self, event):
        x0 = self._val_to_x(self.start_val)
        x1 = self._val_to_x(self.end_val)
        d_start = abs(event.x - x0)
        d_end = abs(event.x - x1)
        if d_start <= d_end and d_start < 20:
            self._dragging = "start"
        elif d_end < 20:
            self._dragging = "end"

    def _on_drag(self, event):
        if not self._dragging:
            return
        val = self._x_to_val(event.x)
        if self._dragging == "start":
            self.start_val = min(val, self.end_val - 0.1)
        else:
            self.end_val = max(val, self.start_val + 0.1)
        self._sync_entries()
        self._draw_slider()

    def _on_release(self, event):
        if self._dragging == "start":
            self._show_frame_at(self.start_val)
        elif self._dragging == "end":
            self._show_frame_at(self.end_val)
        self._dragging = None
        if self._on_range_change:
            self._on_range_change()

    def _on_start_entry(self, event=None):
        try:
            val = round(_parse_time(self.start_entry.get()), 1)
            self.start_val = max(0.0, min(val, self.end_val - 0.1))
            self._show_frame_at(self.start_val)
        except (ValueError, IndexError):
            pass
        self._sync_entries()
        self._draw_slider()
        if self._on_range_change:
            self._on_range_change()

    def _on_end_entry(self, event=None):
        try:
            val = round(_parse_time(self.end_entry.get()), 1)
            self.end_val = max(self.start_val + 0.1, min(val, self.duration))
            self._show_frame_at(self.end_val)
        except (ValueError, IndexError):
            pass
        self._sync_entries()
        self._draw_slider()
        if self._on_range_change:
            self._on_range_change()

    def _sync_entries(self):
        for entry, val in [(self.start_entry, self.start_val), (self.end_entry, self.end_val)]:
            entry.delete(0, tk.END)
            entry.insert(0, _format_time(val))
        clip_dur = self.end_val - self.start_val
        self._duration_label.configure(text=f"Duration: {_format_time(clip_dur)}")

    # -- Preview playback --

    def _toggle_preview(self):
        if self._preview_proc and self._preview_proc.poll() is None:
            self._stop_preview()
            return
        self._play_preview()

    def _play_preview(self):
        if not self.downloaded_path:
            return
        if not self._video_clip:
            self._reopen_video()
            if not self._video_clip:
                return
        try:
            self._video_clip.get_frame(0)
        except Exception:
            self._reopen_video()
            if not self._video_clip:
                return
        self._stop_preview()
        ffmpeg = _get_ffmpeg_path()
        temp_audio = os.path.join(tempfile.gettempdir(), "videosplice_intro_preview.wav")
        start = self.start_val
        dur = self.end_val - self.start_val
        subprocess.run(
            [ffmpeg, "-hide_banner", "-y", "-ss", f"{start:.1f}",
             "-i", self.downloaded_path, "-t", f"{dur:.1f}", "-vn", temp_audio],
            capture_output=True,
        )
        import time as _time
        self._playback_start_time = _time.time()
        self._playback_active = True
        self._preview_btn.configure(text="Stop", fg="#CC3333")
        if sys.platform == "darwin":
            self._preview_proc = subprocess.Popen(["afplay", temp_audio])
        elif sys.platform == "win32":
            self._preview_proc = subprocess.Popen(
                ["cmd", "/c", "start", "/min", "", temp_audio], shell=False,
            )
        self._tick_playback()
        threading.Thread(target=self._wait_preview, daemon=True).start()

    def _tick_playback(self):
        if not self._playback_active:
            return
        import time as _time
        elapsed = _time.time() - self._playback_start_time
        current_t = self.start_val + elapsed
        if current_t > self.end_val:
            self._stop_preview()
            return
        self._show_frame_at(current_t)
        self._draw_slider()
        self.after(42, self._tick_playback)

    def _wait_preview(self):
        if self._preview_proc:
            self._preview_proc.wait()
        self._playback_active = False
        self._playback_start_time = None
        try:
            self._slider.after(0, self._draw_slider)
            self._preview_btn.after(0, lambda: self._preview_btn.configure(text="Preview", fg=ACCENT))
        except tk.TclError:
            pass

    def _stop_preview(self):
        self._playback_active = False
        self._playback_start_time = None
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.terminate()
            self._preview_proc = None
        self._preview_btn.configure(text="Preview", fg=ACCENT)
        self._draw_slider()


class CaptionEditor(tk.Toplevel):
    def __init__(self, parent, caption_words, on_save):
        super().__init__(parent)
        self.title("Edit Captions")
        self.configure(bg=BG)
        self.geometry("460x500")
        self.minsize(400, 300)
        self._on_save = on_save
        self._rows = []

        toolbar = tk.Frame(self, bg=BG)
        toolbar.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(toolbar, text="Edit words and timestamps. Changes save automatically.",
                 font=("Helvetica", 11), bg=BG, fg=MUTED).pack(side="left")

        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, yscrollincrement=1)
        self._scrollbar = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=BG)

        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 10))
        self._scrollbar.pack(side="right", fill="y", pady=(0, 10))

        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._inner.bind("<Enter>", self._bind_mousewheel)
        self._inner.bind("<Leave>", self._unbind_mousewheel)

        for i, w in enumerate(caption_words):
            self._add_insert_button(i)
            self._add_word_row(w)
        self._add_insert_button(len(caption_words))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.transient(parent)

    def _add_word_row(self, word_data):
        idx = len(self._rows)
        row = tk.Frame(self._inner, bg=BG)
        row.pack(fill="x", pady=2)

        word_var = tk.StringVar(value=word_data["word"])
        start_var = tk.StringVar(value=f"{word_data['start']:.2f}")
        end_var = tk.StringVar(value=f"{word_data['end']:.2f}")

        word_entry = tk.Entry(row, textvariable=word_var, width=20,
                              bg=ENTRY_BG, fg=FG, font=("Helvetica", 12),
                              relief="solid", bd=1, insertbackground=FG)
        word_entry.pack(side="left", padx=(0, 4))

        tk.Label(row, text="start:", font=("Helvetica", 10), bg=BG, fg=MUTED).pack(side="left")
        start_entry = tk.Entry(row, textvariable=start_var, width=6,
                               bg=ENTRY_BG, fg=FG, font=("Helvetica", 10),
                               relief="solid", bd=1, insertbackground=FG)
        start_entry.pack(side="left", padx=(2, 4))

        tk.Label(row, text="end:", font=("Helvetica", 10), bg=BG, fg=MUTED).pack(side="left")
        end_entry = tk.Entry(row, textvariable=end_var, width=6,
                             bg=ENTRY_BG, fg=FG, font=("Helvetica", 10),
                             relief="solid", bd=1, insertbackground=FG)
        end_entry.pack(side="left", padx=(2, 4))

        del_btn = tk.Label(row, text="x", font=("Helvetica", 11, "bold"),
                           bg=BG, fg="#CC4444", cursor="hand2")
        del_btn.pack(side="left", padx=(4, 0))
        del_btn.bind("<Button-1>", lambda e, r=idx: self._delete_row(r))

        word_var.trace_add("write", lambda *_: self._auto_save())
        start_var.trace_add("write", lambda *_: self._auto_save())
        end_var.trace_add("write", lambda *_: self._auto_save())

        self._rows.append({
            "frame": row, "word": word_var,
            "start": start_var, "end": end_var,
        })

    def _add_insert_button(self, position):
        btn_frame = tk.Frame(self._inner, bg=BG)
        btn_frame.pack(fill="x")
        btn = tk.Label(btn_frame, text="+ insert word", font=("Helvetica", 9),
                       bg=BG, fg=ACCENT, cursor="hand2")
        btn.pack(anchor="w", padx=20)
        btn.bind("<Button-1>", lambda e, p=position: self._insert_word(p))

    def _insert_word(self, position):
        if position > 0 and position <= len(self._rows):
            prev = self._rows[position - 1]
            try:
                start = float(prev["end"].get())
            except ValueError:
                start = 0.0
            end = start + 0.3
        elif self._rows:
            end_val = self._rows[0]["start"].get()
            try:
                end = float(end_val)
            except ValueError:
                end = 0.3
            start = max(0.0, end - 0.3)
        else:
            start, end = 0.0, 0.3

        new_word = {"word": "", "start": start, "end": end}

        for widget in self._inner.winfo_children():
            widget.destroy()
        old_rows = self._rows
        self._rows = []

        words = self._collect_words_from(old_rows)
        words.insert(position, new_word)

        for i, w in enumerate(words):
            self._add_insert_button(i)
            self._add_word_row(w)
        self._add_insert_button(len(words))

        self._auto_save()
        if self._rows and position < len(self._rows):
            self._rows[position]["frame"].after(50, lambda: self._rows[position]["frame"].winfo_children()[0].focus_set() if self._rows[position]["frame"].winfo_children() else None)

    def _delete_row(self, idx):
        if idx >= len(self._rows):
            return
        for widget in self._inner.winfo_children():
            widget.destroy()
        old_rows = self._rows
        self._rows = []
        words = self._collect_words_from(old_rows)
        words.pop(idx)
        for i, w in enumerate(words):
            self._add_insert_button(i)
            self._add_word_row(w)
        self._add_insert_button(len(words))
        self._auto_save()

    def _collect_words_from(self, rows):
        words = []
        for r in rows:
            try:
                s = float(r["start"].get())
            except ValueError:
                s = 0.0
            try:
                e = float(r["end"].get())
            except ValueError:
                e = s + 0.3
            words.append({"word": r["word"].get(), "start": s, "end": e})
        return words

    def _auto_save(self):
        words = self._collect_words_from(self._rows)
        self._on_save(words)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        bbox = self._canvas.bbox("all")
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        if content_height <= 0:
            return
        shift = (-event.delta * 2) / content_height
        current = self._canvas.yview()
        new_pos = max(0.0, min(1.0, current[0] + shift))
        self._canvas.yview_moveto(new_pos)

    def _close(self):
        self._auto_save()
        self._canvas.unbind_all("<MouseWheel>")
        self.destroy()


class VideoSpliceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Splice")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.font_title = font.Font(family="Helvetica Neue", size=22, weight="bold")
        self.font_section = font.Font(family="Helvetica Neue", size=13, weight="bold")
        self.font_body = font.Font(family="Helvetica Neue", size=12)
        self.font_small = font.Font(family="Helvetica Neue", size=11)
        self.font_btn = font.Font(family="Helvetica Neue", size=14, weight="bold")
        self.font_log = font.Font(family="Menlo", size=10)

        # -- Shared variables (synced across both tabs via textvariable) --
        self.source_mode = tk.StringVar(value="pinterest")
        self.pinterest_var = tk.StringVar()
        self.folder_var = tk.StringVar()
        self.audio_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Desktop"))
        self.count_var = tk.StringVar(value="1")
        self.clip_length_var = tk.StringVar(value="1.0")
        self.image_length_var = tk.StringVar(value="0.1")
        self.bpm_var = tk.StringVar()
        self.images_only_var = tk.BooleanVar()
        self.videos_only_var = tk.BooleanVar()
        self.force_refresh_var = tk.BooleanVar()
        self.sequence_mode_var = tk.BooleanVar()
        self.landscape_var = tk.BooleanVar()
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_pct_var = tk.StringVar(value="")
        self.warning_var = tk.StringVar(value="")
        self.progress_var.trace_add("write", self._update_progress_label)

        self._pinterest_entries = []
        self._folder_entries = []
        self._clip_length_entries = []
        self._image_length_entries = []
        self._generate_frames = []
        self._generate_labels = []
        self._captioning = False
        self._captioning_dots = 0
        self._current_project_path = None

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=ENTRY_BG, background=FG, thickness=6)

        # --- Tab bar ---
        self._active_tab = "basic"

        tab_bar = tk.Frame(root, bg=TAB_INACTIVE_BG)
        tab_bar.pack(fill="x")

        tab_row = tk.Frame(tab_bar, bg=TAB_INACTIVE_BG)
        tab_row.pack(side="left", padx=(32, 0), pady=(6, 0))

        file_row = tk.Frame(tab_bar, bg=TAB_INACTIVE_BG)
        file_row.pack(side="right", padx=(0, 16), pady=(8, 0))
        for label_text, cmd in [("Open", self._open_project),
                                ("Save", self._save_project),
                                ("Save As", self._save_project_as)]:
            btn = tk.Label(file_row, text=label_text, font=self.font_small,
                           bg=TAB_INACTIVE_BG, fg=MUTED, cursor="hand2")
            btn.pack(side="left", padx=(8, 0))
            btn.bind("<Button-1>", lambda e, c=cmd: c())

        self._basic_tab_btn = tk.Label(
            tab_row, text="  Basic  ", font=self.font_body,
            bg=BG, fg=FG, padx=16, pady=6, cursor="hand2",
        )
        self._basic_tab_btn.pack(side="left")
        self._basic_tab_btn.bind("<Button-1>", lambda e: self._switch_tab("basic"))

        self._adv_tab_btn = tk.Label(
            tab_row, text="  Advanced  ", font=self.font_body,
            bg=TAB_INACTIVE_BG, fg=MUTED, padx=16, pady=6, cursor="hand2",
        )
        self._adv_tab_btn.pack(side="left", padx=(2, 0))
        self._adv_tab_btn.bind("<Button-1>", lambda e: self._switch_tab("advanced"))

        # --- Scrollable content area ---
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0, yscrollincrement=1)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=BG)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.scroll_frame.bind("<Enter>", self._bind_mousewheel)
        self.scroll_frame.bind("<Leave>", self._unbind_mousewheel)

        # --- Build both tabs ---
        self.basic_frame = tk.Frame(self.scroll_frame, bg=BG)
        self.advanced_frame = tk.Frame(self.scroll_frame, bg=BG)

        self._build_basic_tab()
        self._build_advanced_tab()

        self.basic_frame.pack(fill="x")

    # ----------------------------------------------------------------
    # Tab switching
    # ----------------------------------------------------------------

    def _switch_tab(self, name):
        if name == self._active_tab:
            return
        self._active_tab = name
        if name == "basic":
            self.advanced_frame.pack_forget()
            self.basic_frame.pack(fill="x")
            self._basic_tab_btn.configure(bg=BG, fg=FG)
            self._adv_tab_btn.configure(bg=TAB_INACTIVE_BG, fg=MUTED)
        else:
            self.basic_frame.pack_forget()
            self.advanced_frame.pack(fill="x")
            self._basic_tab_btn.configure(bg=TAB_INACTIVE_BG, fg=MUTED)
            self._adv_tab_btn.configure(bg=BG, fg=FG)
        self.canvas.yview_moveto(0)

    # ----------------------------------------------------------------
    # Entry factory
    # ----------------------------------------------------------------

    def _make_entry(self, parent, textvariable=None, width=None, state="normal"):
        kwargs = dict(
            font=self.font_body, bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        if textvariable is not None:
            kwargs["textvariable"] = textvariable
        if width is not None:
            kwargs["width"] = width
        if state != "normal":
            kwargs["state"] = state
        return tk.Entry(parent, **kwargs)

    # ----------------------------------------------------------------
    # Shared section builders (used by both tabs)
    # ----------------------------------------------------------------

    def _build_input_source(self, container, pad_x):
        self._divider(container, pad_x)
        self._section_label(container, "Input Source", pad_x)

        source_row = tk.Frame(container, bg=BG)
        source_row.pack(fill="x", padx=pad_x, pady=(0, 6))

        tk.Radiobutton(
            source_row, text="Pinterest URL", variable=self.source_mode,
            value="pinterest", command=self._toggle_source,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left")

        tk.Radiobutton(
            source_row, text="Local Folder", variable=self.source_mode,
            value="local", command=self._toggle_source,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left", padx=(20, 0))

        pinterest_frame = tk.Frame(container, bg=BG)
        pinterest_frame.pack(fill="x", padx=pad_x, pady=(0, 2))
        pe = self._make_entry(pinterest_frame, textvariable=self.pinterest_var)
        pe.pack(fill="x", ipady=6)
        self._pinterest_entries.append(pe)

        refresh_cb = tk.Checkbutton(
            container, text="Force refresh",
            variable=self.force_refresh_var,
            bg=BG, fg=FG, font=self.font_small, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        )
        refresh_cb.pack(anchor="w", padx=pad_x, pady=(0, 4))
        self._pinterest_entries.append(refresh_cb)

        folder_frame = tk.Frame(container, bg=BG)
        folder_frame.pack(fill="x", padx=pad_x, pady=(0, 8))
        fe = self._make_entry(folder_frame, textvariable=self.folder_var, state="disabled")
        fe.pack(side="left", fill="x", expand=True, ipady=6)
        self._folder_entries.append(fe)
        self._browse_button(folder_frame, self._browse_folder).pack(side="left", padx=(10, 0))

    def _build_output_section(self, container, pad_x):
        self._divider(container, pad_x)
        self._section_label(container, "Output", pad_x)

        output_row = tk.Frame(container, bg=BG)
        output_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        dir_frame = tk.Frame(output_row, bg=BG)
        dir_frame.pack(side="left", fill="x", expand=True)
        tk.Label(dir_frame, text="Directory", font=self.font_small,
                 bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        dir_input_row = tk.Frame(dir_frame, bg=BG)
        dir_input_row.pack(fill="x")
        self._make_entry(dir_input_row, textvariable=self.output_dir_var).pack(
            side="left", fill="x", expand=True, ipady=6)
        self._browse_button(dir_input_row, self._browse_output_dir).pack(
            side="left", padx=(10, 0))

        count_frame = tk.Frame(output_row, bg=BG)
        count_frame.pack(side="left", padx=(20, 0))
        tk.Label(count_frame, text="Count", font=self.font_small,
                 bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self._make_entry(count_frame, textvariable=self.count_var, width=5).pack(ipady=6)

    def _build_generate_section(self, container, pad_x):
        self._divider(container, pad_x)

        gen_frame = tk.Frame(container, bg=BTN_BG, cursor="hand2")
        gen_frame.pack(fill="x", padx=pad_x, pady=(12, 8))
        gen_label = tk.Label(
            gen_frame, text="Generate",
            bg=BTN_BG, fg=BTN_FG, font=self.font_btn, pady=14,
        )
        gen_label.pack(fill="x")
        gen_frame.bind("<Button-1>", lambda e: self._start_generate())
        gen_label.bind("<Button-1>", lambda e: self._start_generate())
        self._generate_frames.append(gen_frame)
        self._generate_labels.append(gen_label)

        tk.Label(
            container, textvariable=self.progress_pct_var,
            font=self.font_small, bg=BG, fg=MUTED, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(4, 2))
        ttk.Progressbar(
            container, variable=self.progress_var,
            maximum=100, mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        ).pack(fill="x", padx=pad_x, pady=(0, 8))

        tk.Label(
            container, textvariable=self.warning_var,
            font=self.font_small, bg=BG, fg="#CC6600",
            anchor="w", wraplength=580, justify="left",
        ).pack(fill="x", padx=pad_x, pady=(0, 4))

    # ----------------------------------------------------------------
    # Basic tab
    # ----------------------------------------------------------------

    def _build_basic_tab(self):
        c = self.basic_frame
        pad_x = 32

        tk.Label(
            c, text="Video Splice", font=self.font_title,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(28, 6))

        seq_row = tk.Frame(c, bg=BG)
        seq_row.pack(fill="x", padx=pad_x, pady=(0, 2))
        tk.Checkbutton(
            seq_row, text="Sequence mode", variable=self.sequence_mode_var,
            command=self._toggle_sequence_mode,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left")
        tk.Label(
            seq_row, text="Sync cuts to the beat of your music.",
            font=self.font_small, bg=BG, fg=MUTED,
        ).pack(side="left", padx=(6, 0))

        self._build_input_source(c, pad_x)

        # --- Audio ---
        self._divider(c, pad_x)
        self._section_label(c, "Audio", pad_x)
        tk.Label(
            c, text="Best results with clips under 30 seconds.",
            font=self.font_small, bg=BG, fg=MUTED, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(0, 4))

        audio_row = tk.Frame(c, bg=BG)
        audio_row.pack(fill="x", padx=pad_x, pady=(0, 4))
        self._make_entry(audio_row, textvariable=self.audio_var).pack(
            side="left", fill="x", expand=True, ipady=6)
        self._browse_button(audio_row, self._browse_audio).pack(side="left", padx=(10, 0))

        bpm_row = tk.Frame(c, bg=BG)
        bpm_row.pack(fill="x", padx=pad_x, pady=(0, 8))
        tk.Label(bpm_row, text="BPM (optional)", font=self.font_small,
                 bg=BG, fg=MUTED).pack(side="left")
        self._make_entry(bpm_row, textvariable=self.bpm_var, width=6).pack(
            side="left", padx=(6, 0), ipady=4)

        self._build_output_section(c, pad_x)

        # --- Options ---
        self._divider(c, pad_x)
        self._section_label(c, "Options", pad_x)

        options_row = tk.Frame(c, bg=BG)
        options_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        tk.Checkbutton(
            options_row, text="Landscape", variable=self.landscape_var,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left")

        self._build_generate_section(c, pad_x)

        tk.Frame(c, bg=BG, height=16).pack(fill="x")

    # ----------------------------------------------------------------
    # Advanced tab
    # ----------------------------------------------------------------

    def _build_advanced_tab(self):
        c = self.advanced_frame
        pad_x = 32

        tk.Label(
            c, text="Video Splice", font=self.font_title,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(28, 6))

        seq_row = tk.Frame(c, bg=BG)
        seq_row.pack(fill="x", padx=pad_x, pady=(0, 2))
        tk.Checkbutton(
            seq_row, text="Sequence mode", variable=self.sequence_mode_var,
            command=self._toggle_sequence_mode,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left")
        tk.Label(
            seq_row, text="Sync cuts to the beat of your music.",
            font=self.font_small, bg=BG, fg=MUTED,
        ).pack(side="left", padx=(6, 0))

        self._build_input_source(c, pad_x)

        self._build_output_section(c, pad_x)

        # --- Options ---
        self._divider(c, pad_x)
        self._section_label(c, "Options", pad_x)

        options_row = tk.Frame(c, bg=BG)
        options_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        tk.Checkbutton(
            options_row, text="Images only", variable=self.images_only_var,
            command=self._toggle_media_filter,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left")

        tk.Checkbutton(
            options_row, text="Videos only", variable=self.videos_only_var,
            command=self._toggle_media_filter,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left", padx=(24, 0))

        tk.Checkbutton(
            options_row, text="Landscape", variable=self.landscape_var,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left", padx=(24, 0))

        # --- Timing ---
        self._divider(c, pad_x)
        self._section_label(c, "Timing", pad_x)

        timing_row = tk.Frame(c, bg=BG)
        timing_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        clip_frame = tk.Frame(timing_row, bg=BG)
        clip_frame.pack(side="left")
        tk.Label(clip_frame, text="Video clip (sec)", font=self.font_small,
                 bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        cle = self._make_entry(clip_frame, textvariable=self.clip_length_var, width=8)
        cle.pack(ipady=6)
        self._clip_length_entries.append(cle)

        img_frame = tk.Frame(timing_row, bg=BG)
        img_frame.pack(side="left", padx=(20, 0))
        tk.Label(img_frame, text="Image (sec)", font=self.font_small,
                 bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        ile = self._make_entry(img_frame, textvariable=self.image_length_var, width=8)
        ile.pack(ipady=6)
        self._image_length_entries.append(ile)

        # --- Intro Clip ---
        self._divider(c, pad_x)
        self._section_label(c, "Intro Clip (optional)", pad_x)
        tk.Label(
            c, text="Experimental feature — results may vary.",
            font=self.font_small, bg=BG, fg=MUTED, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(0, 4))

        self.intro_preview = IntroClipPreview(
            c,
            fonts={"small": self.font_small, "body": self.font_body},
            on_range_change=self._update_overlap_visual,
            on_load_start=self._on_intro_load_start,
            on_load_complete=self._on_intro_load_complete,
        )
        self.intro_preview.pack(fill="x", padx=pad_x, pady=(0, 4))

        self.intro_overlap_var = tk.BooleanVar()
        self.intro_duck_db_var = tk.StringVar(value="-8")
        overlap_row = tk.Frame(c, bg=BG)
        overlap_row.pack(fill="x", padx=pad_x, pady=(0, 2))
        self._intro_overlap_cb = tk.Checkbutton(
            overlap_row, text="Overlap",
            variable=self.intro_overlap_var,
            command=self._toggle_overlap,
            bg=BG, fg=FG, font=self.font_small, activebackground=BG,
            highlightthickness=0, selectcolor=BG, state="disabled",
        )
        self._intro_overlap_cb.pack(side="left")
        tk.Label(
            overlap_row, text="Intro plays over the start of the main video",
            font=self.font_small, bg=BG, fg=MUTED,
        ).pack(side="left", padx=(4, 0))

        duck_row = tk.Frame(c, bg=BG)
        duck_row.pack(fill="x", padx=(pad_x + 24, pad_x), pady=(0, 4))
        self._duck_db_entry = self._make_entry(
            duck_row, textvariable=self.intro_duck_db_var, width=5, state="disabled")
        self._duck_db_entry.pack(side="left", ipady=3)
        tk.Label(duck_row, text="Duck audio (dB)", font=self.font_small,
                 bg=BG, fg=FG).pack(side="left", padx=(6, 0))

        self.intro_captions_var = tk.BooleanVar()
        caption_row = tk.Frame(c, bg=BG)
        caption_row.pack(fill="x", padx=pad_x, pady=(0, 8))
        self._intro_captions_cb = tk.Checkbutton(
            caption_row, text="Auto captions",
            variable=self.intro_captions_var,
            command=self._toggle_captions,
            bg=BG, fg=FG, font=self.font_small, activebackground=BG,
            highlightthickness=0, selectcolor=BG, state="disabled",
        )
        self._intro_captions_cb.pack(side="left")
        self._captions_status = tk.Label(
            caption_row, text="", font=self.font_small, bg=BG, fg=MUTED,
        )
        self._captions_status.pack(side="left", padx=(6, 0))
        self._captions_edit_btn = tk.Label(
            caption_row, text="Edit", font=self.font_small,
            bg=BG, fg=MUTED, cursor="arrow",
        )
        self._captions_edit_btn.pack(side="left", padx=(6, 0))
        self._captions_edit_btn.bind("<Button-1>", lambda e: self._open_caption_editor())
        self._caption_editor_window = None

        # --- Audio ---
        self._divider(c, pad_x)
        self._section_label(c, "Audio", pad_x)
        tk.Label(
            c, text="Best results with clips under 30 seconds.",
            font=self.font_small, bg=BG, fg=MUTED, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(0, 4))

        audio_row = tk.Frame(c, bg=BG)
        audio_row.pack(fill="x", padx=pad_x, pady=(0, 4))
        self._make_entry(audio_row, textvariable=self.audio_var).pack(
            side="left", fill="x", expand=True, ipady=6)
        self._browse_button(audio_row, self._browse_audio).pack(side="left", padx=(10, 0))

        self.audio_trimmer = AudioTrimmer(
            c,
            fonts={"small": self.font_small, "body": self.font_body},
            get_overlap_info=self._get_overlap_info,
        )
        self.audio_trimmer.pack(fill="x", padx=pad_x, pady=(0, 4))

        bpm_row = tk.Frame(c, bg=BG)
        bpm_row.pack(fill="x", padx=pad_x, pady=(0, 8))
        tk.Label(bpm_row, text="BPM (optional)", font=self.font_small,
                 bg=BG, fg=MUTED).pack(side="left")
        self._make_entry(bpm_row, textvariable=self.bpm_var, width=6).pack(
            side="left", padx=(6, 0), ipady=4)

        self._build_generate_section(c, pad_x)

        # --- Log ---
        self._divider(c, pad_x)

        tk.Label(
            c, text="Log", font=self.font_section,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(6, 3))

        log_frame = tk.Frame(c, bg=BG)
        log_frame.pack(fill="x", padx=pad_x, pady=(0, 16))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=14, font=self.font_log,
            bg=ENTRY_BG, fg=FG, relief="flat",
            highlightthickness=1, highlightcolor=DIVIDER,
            highlightbackground=DIVIDER, state="disabled",
            wrap="word", insertbackground=FG,
        )
        self.log_text.pack(fill="x")
        self.log_text.bind("<Enter>", self._bind_log_mousewheel)
        self.log_text.bind("<Leave>", self._unbind_log_mousewheel)

    # ----------------------------------------------------------------
    # Canvas / scroll helpers
    # ----------------------------------------------------------------

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_canvas_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_canvas_mousewheel(self, event):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        if content_height <= 0:
            return
        shift = (-event.delta * 2) / content_height
        current = self.canvas.yview()
        new_pos = max(0.0, min(1.0, current[0] + shift))
        self.canvas.yview_moveto(new_pos)

    def _bind_log_mousewheel(self, event):
        self._log_user_scrolled = True
        self.log_text.bind_all("<MouseWheel>", self._on_log_mousewheel)

    def _unbind_log_mousewheel(self, event):
        self._log_user_scrolled = False
        self.log_text.unbind_all("<MouseWheel>")

    def _on_log_mousewheel(self, event):
        self._log_user_scrolled = True
        current = self.log_text.yview()
        total_lines = int(self.log_text.index("end-1c").split(".")[0])
        if total_lines <= 1:
            return
        line_fraction = 1.0 / total_lines
        shift = -event.delta * line_fraction * 0.4
        new_pos = max(0.0, min(1.0, current[0] + shift))
        self.log_text.yview_moveto(new_pos)

    # ----------------------------------------------------------------
    # Widget helpers
    # ----------------------------------------------------------------

    def _browse_button(self, parent, command):
        frame = tk.Frame(parent, bg=DIVIDER, padx=1, pady=1)
        label = tk.Label(
            frame, text="Browse", font=self.font_small,
            bg=BG, fg=FG, padx=10, pady=2, cursor="hand2",
        )
        label.pack()
        label.bind("<Button-1>", lambda e: command())
        frame.bind("<Button-1>", lambda e: command())
        return frame

    def _divider(self, parent, pad_x):
        tk.Frame(parent, bg=DIVIDER, height=1).pack(fill="x", padx=pad_x, pady=(0, 0))

    def _section_label(self, parent, text, pad_x):
        tk.Label(
            parent, text=text, font=self.font_section,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(10, 6))

    # ----------------------------------------------------------------
    # Toggles
    # ----------------------------------------------------------------

    def _toggle_source(self):
        if self.source_mode.get() == "pinterest":
            for entry in self._pinterest_entries:
                entry.configure(state="normal")
            for entry in self._folder_entries:
                entry.configure(state="disabled")
        else:
            for entry in self._pinterest_entries:
                entry.configure(state="disabled")
            for entry in self._folder_entries:
                entry.configure(state="normal")

    def _toggle_media_filter(self):
        if self.images_only_var.get():
            self.videos_only_var.set(False)
        if self.videos_only_var.get():
            self.images_only_var.set(False)

    def _toggle_sequence_mode(self):
        if self.sequence_mode_var.get():
            for entry in self._clip_length_entries:
                entry.configure(state="disabled")
            for entry in self._image_length_entries:
                entry.configure(state="disabled")
            self.intro_preview.enable()
            self._intro_overlap_cb.configure(state="normal")
            if self.intro_preview.downloaded_path:
                self._intro_captions_cb.configure(state="normal")
        else:
            for entry in self._clip_length_entries:
                entry.configure(state="normal")
            for entry in self._image_length_entries:
                entry.configure(state="normal")
            self.intro_preview.disable()
            self._intro_overlap_cb.configure(state="disabled")
            self.intro_overlap_var.set(False)
            self._duck_db_entry.configure(state="disabled")
            self._intro_captions_cb.configure(state="disabled")
            self.intro_captions_var.set(False)
            self.intro_preview._caption_words = []
            self._captions_status.configure(text="")
            self._enable_caption_edit(False)
            self._close_caption_editor()
            self.audio_trimmer.set_overlap_duration(0)

    def _toggle_overlap(self):
        if self.intro_overlap_var.get():
            self._duck_db_entry.configure(state="normal")
            self._update_overlap_visual()
        else:
            self._duck_db_entry.configure(state="disabled")
            self.audio_trimmer.set_overlap_duration(0)

    def _on_intro_load_start(self):
        self.audio_trimmer.set_overlap_duration(0)
        self.intro_overlap_var.set(False)
        self._duck_db_entry.configure(state="disabled")
        self.intro_captions_var.set(False)
        self._intro_captions_cb.configure(state="disabled")
        self.intro_preview._caption_words = []
        self._captions_status.configure(text="")
        self._enable_caption_edit(False)
        self._close_caption_editor()

    def _on_intro_load_complete(self):
        if self.sequence_mode_var.get():
            self._intro_captions_cb.configure(state="normal")

    def _toggle_captions(self):
        if self.intro_captions_var.get():
            if not self.intro_preview.downloaded_path:
                self.intro_captions_var.set(False)
                return
            self._captions_status.configure(text="Transcribing...")
            self._captioning_dots = 0
            self._captioning = True
            self._animate_captioning()
            self.root.configure(cursor="watch")
            self._set_inputs_state("disabled")
            threading.Thread(target=self._run_transcription, daemon=True).start()
        else:
            self.intro_preview._caption_words = []
            self._captions_status.configure(text="")
            self._enable_caption_edit(False)
            self._close_caption_editor()
            if self.intro_preview._video_clip:
                self.intro_preview._show_frame_at(self.intro_preview.start_val)

    def _animate_captioning(self):
        if not self._captioning:
            return
        self._captioning_dots = (self._captioning_dots + 1) % 4
        self._captions_status.configure(
            text="Transcribing" + "." * self._captioning_dots)
        self.root.after(400, self._animate_captioning)

    def _set_inputs_state(self, state):
        for child in self.advanced_frame.winfo_children():
            self._set_widget_state_recursive(child, state)

    def _set_widget_state_recursive(self, widget, state):
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)

    def _run_transcription(self):
        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            from faster_whisper import WhisperModel
            model = WhisperModel("base", compute_type="int8")
            segments, _ = model.transcribe(
                self.intro_preview.downloaded_path, word_timestamps=True)
            words = []
            for seg in segments:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                    })
            self.root.after(0, lambda: self._on_transcription_done(words))
        except Exception as e:
            self.root.after(0, lambda: self._on_transcription_done(None, str(e)))

    def _on_transcription_done(self, words, error=None):
        self._captioning = False
        self.root.configure(cursor="")
        self._set_inputs_state("normal")
        self._toggle_sequence_mode()
        if self.intro_preview.downloaded_path:
            self._intro_captions_cb.configure(state="normal")
        if error:
            self._captions_status.configure(text=f"Error: {error}")
            self.intro_captions_var.set(False)
            return
        if not words:
            self._captions_status.configure(text="No speech detected")
            self.intro_captions_var.set(False)
            return
        self.intro_preview._caption_words = words
        self._captions_status.configure(text=f"{len(words)} words detected")
        self._enable_caption_edit(True)
        if self.intro_preview._video_clip:
            self.intro_preview._show_frame_at(self.intro_preview.start_val)

    def _enable_caption_edit(self, enabled):
        if enabled:
            self._captions_edit_btn.configure(fg=ACCENT, cursor="hand2")
        else:
            self._captions_edit_btn.configure(fg=MUTED, cursor="arrow")

    def _close_caption_editor(self):
        if self._caption_editor_window and self._caption_editor_window.winfo_exists():
            self._caption_editor_window.destroy()
        self._caption_editor_window = None

    def _open_caption_editor(self):
        if not self.intro_preview._caption_words:
            return
        if self._caption_editor_window and self._caption_editor_window.winfo_exists():
            self._caption_editor_window.lift()
            return
        start = self.intro_preview.start_val
        end = self.intro_preview.end_val
        in_range = [w for w in self.intro_preview._caption_words
                    if w["end"] > start and w["start"] < end]
        self._caption_edit_range = (start, end)
        self._caption_editor_window = CaptionEditor(
            self.root, in_range, self._on_captions_edited)

    def _on_captions_edited(self, words):
        start, end = self._caption_edit_range
        out_of_range = [w for w in self.intro_preview._caption_words
                        if w["end"] <= start or w["start"] >= end]
        self.intro_preview._caption_words = out_of_range + words
        self.intro_preview._caption_words.sort(key=lambda w: w["start"])
        count = len([w for w in self.intro_preview._caption_words if w["word"].strip()])
        self._captions_status.configure(text=f"{count} words detected")
        if self.intro_preview._video_clip:
            self.intro_preview._show_frame_at(self.intro_preview.start_val)

    def _update_overlap_visual(self):
        if self.intro_overlap_var.get() and self.intro_preview.duration > 0:
            intro_dur = self.intro_preview.end_val - self.intro_preview.start_val
            self.audio_trimmer.set_overlap_duration(intro_dur)
        else:
            self.audio_trimmer.set_overlap_duration(0)

    def _get_overlap_info(self):
        if not self.intro_overlap_var.get():
            return None
        if not self.intro_preview.downloaded_path:
            return None
        try:
            duck_db = float(self.intro_duck_db_var.get())
        except ValueError:
            duck_db = -8.0
        return (
            self.intro_preview.downloaded_path,
            self.intro_preview.start_val,
            self.intro_preview.end_val,
            duck_db,
        )

    # ----------------------------------------------------------------
    # Browse dialogs
    # ----------------------------------------------------------------

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select media folder")
        if path:
            self.folder_var.set(path)

    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.wav"), ("All files", "*.*")],
        )
        if path:
            self.audio_var.set(path)
            self._load_audio_duration(path)

    def _load_audio_duration(self, path):
        try:
            from moviepy import AudioFileClip
            clip = AudioFileClip(path)
            duration = clip.duration
            clip.close()
            self.audio_trimmer.set_audio(path, duration)
        except Exception:
            pass

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    def _update_progress_label(self, *_args):
        pct = self.progress_var.get()
        if pct <= 0:
            self.progress_pct_var.set("")
        else:
            self.progress_pct_var.set(f"{int(pct)}%")

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------

    def _log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        if not getattr(self, "_log_user_scrolled", False):
            self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    # ----------------------------------------------------------------
    # Build arguments / generate
    # ----------------------------------------------------------------

    def _build_arguments(self):
        if self.source_mode.get() == "pinterest":
            pinterest_url = self.pinterest_var.get().strip()
            if not pinterest_url:
                messagebox.showerror("Error", "Please enter a Pinterest URL.")
                return None
            pinterest = pinterest_url
            input_folder = None
        else:
            input_folder = self.folder_var.get().strip()
            if not input_folder or not os.path.isdir(input_folder):
                messagebox.showerror("Error", "Please select a valid media folder.")
                return None
            pinterest = None

        audio = self.audio_var.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showerror("Error", "Please select a valid audio file.")
            return None

        output_dir = self.output_dir_var.get().strip()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return None

        try:
            count = int(self.count_var.get())
            if count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Count must be a whole number of 1 or more.")
            return None

        sequence_mode = self.sequence_mode_var.get()

        if sequence_mode:
            clip_length = 5.0
            image_length = 3.0
        else:
            try:
                clip_length = float(self.clip_length_var.get())
                if clip_length <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Clip length must be a positive number.")
                return None
            try:
                image_length = float(self.image_length_var.get())
                if image_length <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Image length must be a positive number.")
                return None

        bpm_text = self.bpm_var.get().strip()
        bpm = None
        if bpm_text:
            try:
                bpm = float(bpm_text)
                if bpm <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "BPM must be a positive number.")
                return None

        intro_url = self.intro_preview.url_entry.get().strip() if sequence_mode else ""
        intro_start_sec = self.intro_preview.start_val if intro_url else None
        intro_end_sec = self.intro_preview.end_val if intro_url else None
        intro_video_path = self.intro_preview.downloaded_path if intro_url else None

        audio_start = None
        audio_end = None
        if self.audio_trimmer.duration > 0:
            if self.audio_trimmer.start_val > 0:
                audio_start = self.audio_trimmer.start_val
            if self.audio_trimmer.end_val < self.audio_trimmer.duration:
                audio_end = self.audio_trimmer.end_val

        return Namespace(
            input=input_folder,
            pinterest=pinterest,
            audio=audio,
            output_dir=output_dir,
            count=count,
            clip_length=clip_length,
            image_length=image_length,
            images_only=self.images_only_var.get(),
            videos_only=self.videos_only_var.get(),
            sequence_mode=sequence_mode,
            kick_only=True,
            landscape=self.landscape_var.get(),
            intro_url=intro_url or None,
            intro_start=_format_time(intro_start_sec) if intro_start_sec is not None else "0:00",
            intro_end=_format_time(intro_end_sec) if intro_end_sec is not None else None,
            intro_video_path=intro_video_path,
            intro_overlap=self.intro_overlap_var.get() if intro_url else False,
            duck_db=-abs(float(self.intro_duck_db_var.get())) if self.intro_overlap_var.get() and intro_url else -8.0,
            intro_caption_words=self.intro_preview._caption_words if intro_url and self.intro_captions_var.get() else [],
            bpm=bpm,
            audio_start=audio_start,
            audio_end=audio_end,
            force_refresh=self.force_refresh_var.get(),
        )

    def _collect_project_settings(self):
        return {
            "source_mode": self.source_mode.get(),
            "pinterest_url": self.pinterest_var.get(),
            "input_folder": self.folder_var.get(),
            "audio": self.audio_var.get(),
            "output_dir": self.output_dir_var.get(),
            "count": self.count_var.get(),
            "clip_length": self.clip_length_var.get(),
            "image_length": self.image_length_var.get(),
            "bpm": self.bpm_var.get(),
            "images_only": self.images_only_var.get(),
            "videos_only": self.videos_only_var.get(),
            "sequence_mode": self.sequence_mode_var.get(),
            "landscape": self.landscape_var.get(),
            "intro_url": self.intro_preview.url_entry.get(),
            "intro_start": self.intro_preview.start_val,
            "intro_end": self.intro_preview.end_val,
            "intro_overlap": self.intro_overlap_var.get(),
            "duck_db": self.intro_duck_db_var.get(),
            "audio_start": self.audio_trimmer.start_val if self.audio_trimmer.duration > 0 else None,
            "audio_end": self.audio_trimmer.end_val if self.audio_trimmer.duration > 0 else None,
            "force_refresh": self.force_refresh_var.get(),
            "captions_enabled": self.intro_captions_var.get(),
            "caption_words": self.intro_preview._caption_words if self.intro_captions_var.get() else [],
        }

    def _apply_project_settings(self, data):
        self.source_mode.set(data.get("source_mode", "pinterest"))
        self.pinterest_var.set(data.get("pinterest_url", ""))
        self.folder_var.set(data.get("input_folder", ""))
        self.audio_var.set(data.get("audio", ""))
        self.output_dir_var.set(data.get("output_dir", str(Path.home() / "Desktop")))
        self.count_var.set(data.get("count", "1"))
        self.clip_length_var.set(data.get("clip_length", "1.0"))
        self.image_length_var.set(data.get("image_length", "0.1"))
        self.bpm_var.set(data.get("bpm", ""))
        self.images_only_var.set(data.get("images_only", False))
        self.videos_only_var.set(data.get("videos_only", False))
        self.sequence_mode_var.set(data.get("sequence_mode", False))
        self.landscape_var.set(data.get("landscape", False))
        self.force_refresh_var.set(data.get("force_refresh", False))
        self.intro_overlap_var.set(data.get("intro_overlap", False))
        self.intro_duck_db_var.set(data.get("duck_db", "-8"))

        self._toggle_sequence_mode()
        self._toggle_source()

        if data.get("intro_overlap"):
            self._duck_db_entry.configure(state="normal")

        audio_path = data.get("audio", "")
        if audio_path and os.path.isfile(audio_path):
            self._load_audio_duration(audio_path)
            a_start = data.get("audio_start")
            a_end = data.get("audio_end")

            def _restore_audio_range():
                if self.audio_trimmer.duration > 0:
                    if a_start is not None:
                        self.audio_trimmer.start_val = a_start
                    if a_end is not None:
                        self.audio_trimmer.end_val = a_end
                    self.audio_trimmer._sync_entries()
                    self.audio_trimmer._draw()
                else:
                    self.root.after(100, _restore_audio_range)
            if a_start is not None or a_end is not None:
                self.root.after(200, _restore_audio_range)

        intro_url = data.get("intro_url", "").strip()
        if intro_url:
            self.intro_preview.url_entry.delete(0, "end")
            self.intro_preview.url_entry.insert(0, intro_url)

        caption_words = data.get("caption_words", [])
        if caption_words:
            self.intro_preview._caption_words = caption_words
            self.intro_captions_var.set(True)
            self._captions_status.configure(text=f"{len(caption_words)} words detected")
            self._enable_caption_edit(True)

    def _save_project(self):
        if self._current_project_path:
            self._write_project_file(self._current_project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".vsproj",
            filetypes=[("Video Splice Project", "*.vsproj")],
            title="Save Project",
        )
        if not path:
            return
        self._write_project_file(path)

    def _write_project_file(self, path):
        data = self._collect_project_settings()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._current_project_path = path
        self.root.title(f"Video Splice — {os.path.basename(path)}")
        messagebox.showinfo("Saved", f"Project saved to {os.path.basename(path)}")

    def _open_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Splice Project", "*.vsproj")],
            title="Open Project",
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            messagebox.showerror("Error", f"Could not open project file:\n{e}")
            return
        self._apply_project_settings(data)
        self._current_project_path = path
        self.root.title(f"Video Splice — {os.path.basename(path)}")

    def _start_generate(self):
        arguments = self._build_arguments()
        if arguments is None:
            return

        for label in self._generate_labels:
            label.configure(text="Generating...", fg=BTN_DISABLED_FG)
        for frame in self._generate_frames:
            frame.unbind("<Button-1>")
        for label in self._generate_labels:
            label.unbind("<Button-1>")
        self.progress_var.set(0.0)
        self.warning_var.set("")
        self._log_user_scrolled = False
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

        cli_parts = ["python video_splice.py"]
        if arguments.pinterest:
            cli_parts.append(f'-p "{arguments.pinterest}"')
        else:
            cli_parts.append(f'-i "{arguments.input}"')
        cli_parts.append(f'-a "{arguments.audio}"')
        if arguments.audio_start is not None:
            cli_parts.append(f'--audio-start {arguments.audio_start}')
        if arguments.audio_end is not None:
            cli_parts.append(f'--audio-end {arguments.audio_end}')
        cli_parts.append(f'-o "{arguments.output_dir}"')
        if arguments.count > 1:
            cli_parts.append(f'--count {arguments.count}')
        if arguments.sequence_mode:
            cli_parts.append('--sequence-mode')
        else:
            cli_parts.append(f'--clip-length {arguments.clip_length}')
            cli_parts.append(f'--image-length {arguments.image_length}')
        if arguments.bpm:
            cli_parts.append(f'--bpm {arguments.bpm}')
        if arguments.landscape:
            cli_parts.append('--landscape')
        if arguments.intro_url:
            cli_parts.append(f'--intro-url "{arguments.intro_url}"')
            cli_parts.append(f'--intro-start {arguments.intro_start}')
            cli_parts.append(f'--intro-end {arguments.intro_end}')
            if arguments.intro_overlap:
                cli_parts.append('--intro-overlap')
        if arguments.images_only:
            cli_parts.append('--images-only')
        elif arguments.videos_only:
            cli_parts.append('--videos-only')
        cli_command = " ".join(cli_parts)
        self._log(f"CLI command:\n{cli_command}\n")

        thread = threading.Thread(target=self._run_pipeline_thread, args=(arguments,), daemon=True)
        thread.start()

    def _poll_log_queue(self):
        import queue as _queue
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log(msg)
                if msg.startswith("WARNING:"):
                    self.warning_var.set(msg)
        except _queue.Empty:
            pass
        if self._pipeline_running:
            self.root.after(100, self._poll_log_queue)

    def _run_pipeline_thread(self, arguments):
        import io
        import contextlib
        import queue

        self._log_queue = queue.Queue()
        self._pipeline_running = True
        self.root.after(100, self._poll_log_queue)

        class QueueWriter:
            def __init__(self, q):
                self._queue = q
                self._buffer = ""
            def write(self, text):
                self._buffer += text
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    self._queue.put(line)
            def flush(self):
                if self._buffer:
                    self._queue.put(self._buffer)
                    self._buffer = ""

        writer = QueueWriter(self._log_queue)

        def _on_progress(current, total):
            pct = (current / total) * 100 if total > 0 else 0
            self.root.after(0, self.progress_var.set, pct)

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                run_pipeline(arguments, progress_callback=_on_progress)
            writer.flush()
            self.root.after(0, lambda: messagebox.showinfo("Done", f"Generated {arguments.count} video(s) in:\n{arguments.output_dir}"))
        except Exception as e:
            import traceback
            writer.flush()
            err_msg = str(e)
            err_trace = traceback.format_exc()
            self._log_queue.put(f"\nERROR: {err_msg}\n\n{err_trace}")
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self._pipeline_running = False
            def _re_enable():
                for label in self._generate_labels:
                    label.configure(text="Generate", fg=BTN_FG)
                for frame in self._generate_frames:
                    frame.bind("<Button-1>", lambda e: self._start_generate())
                for label in self._generate_labels:
                    label.bind("<Button-1>", lambda e: self._start_generate())
            self.root.after(0, _re_enable)


def main():
    root = tk.Tk()
    root.geometry("820x800")
    root.minsize(600, 500)
    VideoSpliceUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
