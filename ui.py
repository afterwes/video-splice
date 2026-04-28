"""
ui.py — Tkinter GUI for Video Splice.

Launch with: python ui.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import threading
import tkinter as tk
from argparse import Namespace
from tkinter import filedialog, messagebox, scrolledtext, font, ttk

from video_splice import run_pipeline

BG = "#F0EDE8"
FG = "#1A1A1A"
MUTED = "#999999"
DIVIDER = "#D4D0CA"
ENTRY_BG = "#FFFFFF"
BTN_BG = "#1A1A1A"
BTN_FG = "#F0EDE8"
BTN_DISABLED_FG = "#666666"


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

        pad_x = 32

        # Scrollable content area
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
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

        container = self.scroll_frame

        # --- Title ---
        tk.Label(
            container, text="Video Splice", font=self.font_title,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(28, 4))

        tk.Label(
            container, text="Randomly splice media into vertical videos with audio",
            font=self.font_small, bg=BG, fg=MUTED, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(0, 12))

        # --- Input Source ---
        self._divider(container, pad_x)
        self._section_label(container, "Input Source", pad_x)

        source_row = tk.Frame(container, bg=BG)
        source_row.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.source_mode = tk.StringVar(value="pinterest")

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

        # Pinterest entry
        self.pinterest_frame = tk.Frame(container, bg=BG)
        self.pinterest_frame.pack(fill="x", padx=pad_x, pady=(0, 4))
        self.pinterest_entry = tk.Entry(
            self.pinterest_frame, font=self.font_body, bg=ENTRY_BG,
            fg=FG, relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.pinterest_entry.pack(fill="x", ipady=6)

        # Local folder entry
        self.folder_frame = tk.Frame(container, bg=BG)
        self.folder_frame.pack(fill="x", padx=pad_x, pady=(0, 8))
        self.folder_entry = tk.Entry(
            self.folder_frame, font=self.font_body, bg=ENTRY_BG,
            fg=FG, relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm", state="disabled",
        )
        self.folder_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.browse_folder_button = self._browse_button(self.folder_frame, self._browse_folder)
        self.browse_folder_button.pack(side="left", padx=(10, 0))

        # --- Audio ---
        self._divider(container, pad_x)
        self._section_label(container, "Audio", pad_x)

        audio_row = tk.Frame(container, bg=BG)
        audio_row.pack(fill="x", padx=pad_x, pady=(0, 8))
        self.audio_entry = tk.Entry(
            audio_row, font=self.font_body, bg=ENTRY_BG,
            fg=FG, relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.audio_entry.pack(side="left", fill="x", expand=True, ipady=6)
        audio_browse = self._browse_button(audio_row, self._browse_audio)
        audio_browse.pack(side="left", padx=(10, 0))

        # --- Output ---
        self._divider(container, pad_x)
        self._section_label(container, "Output", pad_x)

        output_row = tk.Frame(container, bg=BG)
        output_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        dir_frame = tk.Frame(output_row, bg=BG)
        dir_frame.pack(side="left", fill="x", expand=True)
        tk.Label(dir_frame, text="Directory", font=self.font_small, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        dir_input_row = tk.Frame(dir_frame, bg=BG)
        dir_input_row.pack(fill="x")
        self.output_dir_entry = tk.Entry(
            dir_input_row, font=self.font_body, bg=ENTRY_BG,
            fg=FG, relief="flat", highlightthickness=1,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.output_dir_entry.insert(0, ".")
        self.output_dir_entry.pack(side="left", fill="x", expand=True, ipady=6)
        output_browse = self._browse_button(dir_input_row, self._browse_output_dir)
        output_browse.pack(side="left", padx=(10, 0))

        count_frame = tk.Frame(output_row, bg=BG)
        count_frame.pack(side="left", padx=(20, 0))
        tk.Label(count_frame, text="Count", font=self.font_small, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self.count_entry = tk.Entry(
            count_frame, font=self.font_body, bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=5,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.count_entry.insert(0, "1")
        self.count_entry.pack(ipady=6)

        # --- Timing ---
        self._divider(container, pad_x)
        self._section_label(container, "Timing", pad_x)

        timing_row = tk.Frame(container, bg=BG)
        timing_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        clip_frame = tk.Frame(timing_row, bg=BG)
        clip_frame.pack(side="left")
        tk.Label(clip_frame, text="Video clip (sec)", font=self.font_small, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self.clip_length_entry = tk.Entry(
            clip_frame, font=self.font_body, bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=8,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.clip_length_entry.insert(0, "1.0")
        self.clip_length_entry.pack(ipady=6)

        img_frame = tk.Frame(timing_row, bg=BG)
        img_frame.pack(side="left", padx=(20, 0))
        tk.Label(img_frame, text="Image (sec)", font=self.font_small, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self.image_length_entry = tk.Entry(
            img_frame, font=self.font_body, bg=ENTRY_BG, fg=FG,
            relief="flat", highlightthickness=1, width=8,
            highlightcolor=DIVIDER, highlightbackground=DIVIDER,
            insertbackground=FG, cursor="xterm",
        )
        self.image_length_entry.insert(0, "0.1")
        self.image_length_entry.pack(ipady=6)

        # --- Options ---
        self._divider(container, pad_x)
        self._section_label(container, "Options", pad_x)

        options_row = tk.Frame(container, bg=BG)
        options_row.pack(fill="x", padx=pad_x, pady=(0, 8))

        self.images_only_var = tk.BooleanVar()
        self.videos_only_var = tk.BooleanVar()
        self.sequence_mode_var = tk.BooleanVar()
        self.landscape_var = tk.BooleanVar()

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
            options_row, text="Sequence mode", variable=self.sequence_mode_var,
            command=self._toggle_sequence_mode,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left", padx=(24, 0))

        tk.Checkbutton(
            options_row, text="Landscape", variable=self.landscape_var,
            bg=BG, fg=FG, font=self.font_body, activebackground=BG,
            highlightthickness=0, selectcolor=BG,
        ).pack(side="left", padx=(24, 0))


        # --- Generate Button ---
        self._divider(container, pad_x)

        self.generate_frame = tk.Frame(container, bg=BTN_BG, cursor="hand2")
        self.generate_frame.pack(fill="x", padx=pad_x, pady=(12, 8))

        self.generate_label = tk.Label(
            self.generate_frame, text="Generate",
            bg=BTN_BG, fg=BTN_FG, font=self.font_btn, pady=14,
        )
        self.generate_label.pack(fill="x")

        self.generate_frame.bind("<Button-1>", lambda e: self._start_generate())
        self.generate_label.bind("<Button-1>", lambda e: self._start_generate())

        # --- Progress Bar ---
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=ENTRY_BG, background=FG, thickness=6)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            container, variable=self.progress_var,
            maximum=100, mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x", padx=pad_x, pady=(0, 8))

        # --- Log ---
        self._divider(container, pad_x)

        tk.Label(
            container, text="Log", font=self.font_section,
            bg=BG, fg=FG, anchor="w",
        ).pack(fill="x", padx=pad_x, pady=(6, 3))

        log_frame = tk.Frame(container, bg=BG)
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

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_canvas_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_canvas_mousewheel(self, event):
        self.canvas.yview_scroll(-event.delta, "units")

    def _bind_log_mousewheel(self, event):
        self.log_text.bind_all("<MouseWheel>", self._on_log_mousewheel)

    def _unbind_log_mousewheel(self, event):
        self.log_text.unbind_all("<MouseWheel>")

    def _on_log_mousewheel(self, event):
        self.log_text.yview_scroll(-event.delta, "units")

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

    def _toggle_source(self):
        if self.source_mode.get() == "pinterest":
            self.pinterest_entry.configure(state="normal")
            self.folder_entry.configure(state="disabled")
        else:
            self.pinterest_entry.configure(state="disabled")
            self.folder_entry.configure(state="normal")

    def _toggle_media_filter(self):
        if self.images_only_var.get():
            self.videos_only_var.set(False)
        if self.videos_only_var.get():
            self.images_only_var.set(False)

    def _toggle_sequence_mode(self):
        if self.sequence_mode_var.get():
            self.clip_length_entry.configure(state="disabled")
            self.image_length_entry.configure(state="disabled")
        else:
            self.clip_length_entry.configure(state="normal")
            self.image_length_entry.configure(state="normal")

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select media folder")
        if path:
            self.folder_entry.configure(state="normal")
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, path)

    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.wav"), ("All files", "*.*")],
        )
        if path:
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, path)

    def _log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _build_arguments(self):
        if self.source_mode.get() == "pinterest":
            pinterest_url = self.pinterest_entry.get().strip()
            if not pinterest_url:
                messagebox.showerror("Error", "Please enter a Pinterest URL.")
                return None
            pinterest = pinterest_url
            input_folder = None
        else:
            input_folder = self.folder_entry.get().strip()
            if not input_folder or not os.path.isdir(input_folder):
                messagebox.showerror("Error", "Please select a valid media folder.")
                return None
            pinterest = None

        audio = self.audio_entry.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showerror("Error", "Please select a valid audio file.")
            return None

        output_dir = self.output_dir_entry.get().strip()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return None

        try:
            count = int(self.count_entry.get())
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
                clip_length = float(self.clip_length_entry.get())
                if clip_length <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Clip length must be a positive number.")
                return None
            try:
                image_length = float(self.image_length_entry.get())
                if image_length <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Image length must be a positive number.")
                return None

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
        )

    def _start_generate(self):
        arguments = self._build_arguments()
        if arguments is None:
            return

        self.generate_label.configure(text="Generating...", fg=BTN_DISABLED_FG)
        self.generate_frame.unbind("<Button-1>")
        self.generate_label.unbind("<Button-1>")
        self.progress_var.set(0.0)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

        cli_parts = ["python video_splice.py"]
        if arguments.pinterest:
            cli_parts.append(f'-p "{arguments.pinterest}"')
        else:
            cli_parts.append(f'-i "{arguments.input}"')
        cli_parts.append(f'-a "{arguments.audio}"')
        cli_parts.append(f'-o "{arguments.output_dir}"')
        if arguments.count > 1:
            cli_parts.append(f'--count {arguments.count}')
        if arguments.sequence_mode:
            cli_parts.append('--sequence-mode')
        else:
            cli_parts.append(f'--clip-length {arguments.clip_length}')
            cli_parts.append(f'--image-length {arguments.image_length}')
        if arguments.landscape:
            cli_parts.append('--landscape')
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
                self.generate_label.configure(text="Generate", fg=BTN_FG)
                self.generate_frame.bind("<Button-1>", lambda e: self._start_generate())
                self.generate_label.bind("<Button-1>", lambda e: self._start_generate())
            self.root.after(0, _re_enable)


def main():
    root = tk.Tk()
    root.geometry("660x960")
    root.minsize(500, 600)
    VideoSpliceUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
