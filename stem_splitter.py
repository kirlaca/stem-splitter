"""StemSplitter – AI music source separation (BS-RoFormer / Demucs) with a job queue.

Portable: models go to the "models" folder next to the app, ffmpeg to "ffmpeg";
the job queue and settings persist in "jobs.json" / "settings.json".
Uses the GPU automatically when the bundled PyTorch has CUDA and the card is supported.
"""
import os
import sys
import json
import time
import uuid
import queue
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import worker
from worker import MODELS

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

APP_NAME = "StemSplitter"
APP_DIR = worker.app_dir()
RES_DIR = getattr(sys, "_MEIPASS", APP_DIR)
JOBS_FILE = os.path.join(APP_DIR, "jobs.json")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
DEFAULT_OUT = os.path.join(APP_DIR, "output")
AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".aiff", ".aif")
AUDIO_TYPES = [("Audio files", " ".join("*" + e for e in AUDIO_EXT)), ("All files", "*.*")]

WAITING, RUNNING, DONE, FAILED, STOPPED = "waiting", "running", "done", "failed", "stopped"
STATUS_TEXT = {WAITING: "○  Waiting", RUNNING: "▶  Running", DONE: "✔  Done",
               FAILED: "✖  Failed", STOPPED: "■  Stopped"}
QUALITY = {"Fast": (1, 2), "Normal": (2, 4), "Best": (5, 8)}   # (demucs shifts, roformer overlap)
DEVICES = ["Auto (GPU if available)", "CPU only"]

PALETTE = {
    "light": {"muted": "#6b6b6b", "drop": "#f3f3f3", "drop_hi": "#e3eefc", "log_bg": "#fafafa",
              "log_fg": "#202020", "ok": "#0f7b0f", "err": "#c42b1c", "run": "#005fb8", "stop": "#9d5d00"},
    "dark": {"muted": "#a0a0a0", "drop": "#2b2b2b", "drop_hi": "#1f3a5a", "log_bg": "#1c1c1c",
             "log_fg": "#d6d6d6", "ok": "#6ccb5f", "err": "#ff99a4", "run": "#60cdff", "stop": "#fce100"},
}


def fmt_dur(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_root():
    """Tk root with drag & drop support when tkinterdnd2 is available."""
    try:
        from tkinterdnd2 import TkinterDnD
        return TkinterDnD.Tk(), True
    except Exception:
        return tk.Tk(), False


def set_dark_titlebar(root, dark):
    """Windows 10/11: match the native title bar to the theme."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        val = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new / old builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), 4) == 0:
                break
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()   # force a non-client repaint
        root.geometry(f"{w + 1}x{h}")
        root.geometry(f"{w}x{h}")
    except Exception:
        pass


class App:
    def __init__(self, root, dnd):
        self.root = root
        self.dnd = dnd
        root.title(APP_NAME)
        root.geometry("1120x860")
        root.minsize(980, 720)
        icon = os.path.join(RES_DIR, "assets", "icon.png")
        if os.path.isfile(icon):
            self._icon = tk.PhotoImage(file=icon)
            root.iconphoto(True, self._icon)

        self.q = mp.Queue()
        self.proc = None
        self.cur = None
        self.cur_t0 = 0
        self.running = False
        self.jobs = [j for j in load_json(JOBS_FILE, []) if j.get("status") in STATUS_TEXT]
        for j in self.jobs:       # a job that was running when the app closed is queued again
            if j["status"] == RUNNING:
                j["status"] = WAITING
        self.cfg = load_json(SETTINGS_FILE, {})
        self.logs = {}
        self.files = []

        self.theme = self.cfg.get("theme", "dark")
        self._apply_theme(first=True)

        outer = ttk.Frame(root, padding=(16, 12, 16, 8))
        outer.pack(fill="both", expand=True)
        self._build_header(outer)
        self._build_setup(outer)
        self._build_queue(outer)
        self.status = ttk.Label(outer, text="Ready.", style="Muted.TLabel")
        self.status.pack(fill="x", pady=(6, 0))

        self.refresh_tree()
        self._model_changed()
        self._show_files()
        self._recolor()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Delete>", lambda e: self.remove_sel() if self.root.focus_get() == self.tree else None)
        root.after(150, self._poll)
        root.after(50, lambda: set_dark_titlebar(root, self.theme == "dark"))
        if self.cfg.get("autostart") and any(j["status"] == WAITING for j in self.jobs):
            root.after(600, self.start_queue)

    # ------------------------------------------------------------ theme
    def _apply_theme(self, first=False):
        import sv_ttk
        sv_ttk.set_theme(self.theme)
        st = ttk.Style()
        p = PALETTE[self.theme]
        st.configure("Title.TLabel", font=("Segoe UI Semibold", 20))
        st.configure("Sub.TLabel", font=("Segoe UI", 10), foreground=p["muted"])
        st.configure("Muted.TLabel", foreground=p["muted"])
        st.configure("CardTitle.TLabel", font=("Segoe UI Semibold", 11))
        st.configure("Step.TLabel", font=("Segoe UI Semibold", 11), foreground=p["run"])
        st.configure("Small.TLabel", font=("Segoe UI", 9), foreground=p["muted"])
        st.configure("Model.TRadiobutton", font=("Segoe UI Semibold", 10))
        st.configure("Treeview", rowheight=30)
        st.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
        if not first:
            self._recolor()
            set_dark_titlebar(self.root, self.theme == "dark")

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme()
        self._save_settings()

    def _recolor(self):
        p = PALETTE[self.theme]
        self.theme_btn.configure(text="☀  Light" if self.theme == "dark" else "☾  Dark")
        self.drop.configure(background=p["drop"], foreground=p["muted"] if not self.files else p["log_fg"])
        self.log_txt.configure(background=p["log_bg"], foreground=p["log_fg"], insertbackground=p["log_fg"])
        for s, c in [(DONE, p["ok"]), (FAILED, p["err"]), (RUNNING, p["run"]), (STOPPED, p["stop"])]:
            self.tree.tag_configure(s, foreground=c)
        if getattr(self, "mdlg", None) and self.mdlg.winfo_exists():
            self._refresh_models_dialog()

    # ------------------------------------------------------------ layout
    def _card(self, parent, step, title):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10, 14, 12))
        head = ttk.Frame(card)
        head.pack(fill="x", pady=(0, 8))
        if step:
            ttk.Label(head, text=step, style="Step.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(head, text=title, style="CardTitle.TLabel").pack(side="left")
        body = ttk.Frame(card)
        body.pack(fill="both", expand=True)
        return card, body, head

    def _build_header(self, parent):
        h = ttk.Frame(parent)
        h.pack(fill="x", pady=(0, 10))
        left = ttk.Frame(h)
        left.pack(side="left")
        ttk.Label(left, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="AI music source separation  ·  BS-RoFormer & Demucs  ·  GPU used automatically "
                             "when available", style="Sub.TLabel").pack(anchor="w")
        self.theme_btn = ttk.Button(h, text="", width=10, command=self.toggle_theme)
        self.theme_btn.pack(side="right", anchor="n", pady=4)
        self.help_btn = ttk.Button(h, text="?  Help", command=self.help_menu)
        self.help_btn.pack(side="right", anchor="n", pady=4, padx=(8, 0))
        ttk.Button(h, text="⟳  Models", command=self.models_dialog).pack(side="right", anchor="n", pady=4, padx=8)

    def help_menu(self):
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="User guide (English)", command=lambda: self.open_guide("en"))
        m.add_command(label="Felhasználói leírás (magyar)", command=lambda: self.open_guide("hu"))
        b = self.help_btn
        try:
            m.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())
        finally:
            m.grab_release()

    def open_guide(self, lang):
        p = os.path.join(RES_DIR, "docs", f"user_guide_{lang}.html")
        if os.path.isfile(p):
            os.startfile(p)
        else:
            messagebox.showerror("Help", f"User guide not found:\n{p}")

    def _build_setup(self, parent):
        c = self.cfg
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=2, uniform="col")
        grid.columnconfigure(1, weight=3, uniform="col")

        # 1 – input
        card, body, head = self._card(grid, "1", "Input")
        card.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        for txt, cmd in [("Clear", self.clear_files), ("Add folder…", self.pick_folder),
                         ("Add files…", self.pick_files)]:
            ttk.Button(head, text=txt, command=cmd).pack(side="right", padx=(6, 0))
        self.drop = tk.Label(body, text="", height=3, font=("Segoe UI", 10), bd=0, cursor="hand2",
                             justify="center", wraplength=900)
        self.drop.pack(fill="x")
        self.drop.bind("<Button-1>", lambda e: self.pick_files())
        if self.dnd:
            from tkinterdnd2 import DND_FILES
            for w in (self.drop, self.root):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop)
            self.drop.dnd_bind("<<DropEnter>>", lambda e: self._drop_hi(True))
            self.drop.dnd_bind("<<DropLeave>>", lambda e: self._drop_hi(False))

        # 2 – model
        card, body, _ = self._card(grid, "2", "Model")
        card.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.model_var = tk.StringVar(value=c.get("model", "roformer"))
        for key, spec in MODELS.items():
            ttk.Radiobutton(body, text=f"{spec['title']}  ·  {spec['name']}", value=key, variable=self.model_var,
                            style="Model.TRadiobutton", command=self._model_changed).pack(anchor="w", pady=(4, 0))
            ttk.Label(body, text=spec["desc"], style="Small.TLabel").pack(anchor="w", padx=(30, 0))

        # 3 – output
        card, body, _ = self._card(grid, "3", "Output")
        card.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text="Stems").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 12))
        self.stem_box = ttk.Frame(body)
        self.stem_box.grid(row=0, column=1, columnspan=3, sticky="w")
        self.saved_stems = c.get("stems", {})
        self.stem_vars = {}

        ttk.Label(body, text="Folder").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 12))
        self.out_var = tk.StringVar(value=c.get("output_dir", DEFAULT_OUT))
        ttk.Entry(body, textvariable=self.out_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(body, text="…", width=3, command=self.pick_out).grid(row=1, column=3, padx=(6, 0))
        self.sub_var = tk.BooleanVar(value=c.get("subfolder", True))
        ttk.Checkbutton(body, text="Separate folder per song", variable=self.sub_var,
                        style="Switch.TCheckbutton").grid(row=2, column=1, columnspan=3, sticky="w", pady=(0, 4))

        opts = ttk.Frame(body)
        opts.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.fmt_var = tk.StringVar(value=c.get("format", "WAV"))
        self.qual_var = tk.StringVar(value=c.get("quality", "Normal"))
        self.dev_var = tk.StringVar(value=c.get("device", DEVICES[0]))
        if self.qual_var.get() not in QUALITY:
            self.qual_var.set("Normal")
        if self.dev_var.get() not in DEVICES:
            self.dev_var.set(DEVICES[0])
        for i, (lbl, var, vals, w) in enumerate([("Format", self.fmt_var, ["WAV", "FLAC", "MP3"], 6),
                                                 ("Quality", self.qual_var, list(QUALITY), 8),
                                                 ("Device", self.dev_var, DEVICES, 19)]):
            f = ttk.Frame(opts)
            f.pack(side="left", padx=(0, 10))
            ttk.Label(f, text=lbl, style="Small.TLabel").pack(anchor="w")
            ttk.Combobox(f, textvariable=var, values=vals, state="readonly", width=w).pack(anchor="w")
        ttk.Button(opts, text="＋  Add to queue", style="Accent.TButton",
                   command=self.add_jobs).pack(side="right", anchor="s")

    def _build_queue(self, parent):
        pw = ttk.PanedWindow(parent, orient="vertical")
        pw.pack(fill="both", expand=True, pady=(10, 0))

        card, body, head = self._card(pw, "", "Queue")
        pw.add(card, weight=3)
        self.start_btn = ttk.Button(head, text="▶  Start", style="Accent.TButton", command=self.start_queue)
        self.start_btn.pack(side="left", padx=(16, 4))
        self.pause_btn = ttk.Button(head, text="⏸  Pause after current", command=self.pause_queue)
        self.pause_btn.pack(side="left", padx=4)
        self.kill_btn = ttk.Button(head, text="■  Stop current", command=self.kill_current)
        self.kill_btn.pack(side="left", padx=4)
        self.auto_var = tk.BooleanVar(value=self.cfg.get("autostart", False))
        ttk.Checkbutton(head, text="Resume on launch", variable=self.auto_var, style="Switch.TCheckbutton",
                        command=self._save_settings).pack(side="right")

        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(0, 6))
        for txt, cmd in [("↻  Retry", self.retry_sel), ("▲", lambda: self.move_sel(-1)),
                         ("▼", lambda: self.move_sel(1)), ("✕  Remove", self.remove_sel),
                         ("Clear finished", self.clear_finished)]:
            ttk.Button(bar, text=txt, command=cmd, width=3 if len(txt) == 1 else None).pack(side="left", padx=(0, 4))
        ttk.Button(bar, text="📁  Open output", command=self.open_out).pack(side="right")

        tf = ttk.Frame(body)
        tf.pack(fill="both", expand=True)
        cols = ("file", "model", "stems", "fmt", "status", "info")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended", height=6)
        for col, t, w, st in [("file", "File", 250, True), ("model", "Model", 110, False),
                              ("stems", "Stems", 190, False), ("fmt", "Format", 60, False),
                              ("status", "Status", 105, False), ("info", "Details", 280, True)]:
            self.tree.heading(col, text=t, anchor="w")
            self.tree.column(col, width=w, stretch=st, anchor="w")
        sb = ttk.Scrollbar(tf, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.show_log())
        self.tree.bind("<Double-1>", lambda e: self.open_out())

        pr = ttk.Frame(body)
        pr.pack(fill="x", pady=(8, 0))
        self.cur_lbl = ttk.Label(pr, text="Idle", style="Small.TLabel", width=48)
        self.cur_lbl.pack(side="left")
        self.pct_lbl = ttk.Label(pr, text="", style="Small.TLabel", width=5, anchor="e")
        self.pct_lbl.pack(side="right")
        self.progress = ttk.Progressbar(pr, maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        card, body, _ = self._card(pw, "", "Log  ·  selected job")
        pw.add(card, weight=1)
        self.log_txt = tk.Text(body, height=6, wrap="none", font=("Cascadia Mono", 9), bd=0,
                               highlightthickness=0, padx=8, pady=6, state="disabled")
        lsb = ttk.Scrollbar(body, command=self.log_txt.yview)
        self.log_txt.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self.log_txt.pack(fill="both", expand=True)
        self._buttons()

    # ------------------------------------------------------------ input
    def _drop_hi(self, on):
        p = PALETTE[self.theme]
        self.drop.configure(background=p["drop_hi"] if on else p["drop"])

    def _on_drop(self, e):
        self._drop_hi(False)
        self._add_paths(self.root.tk.splitlist(e.data))

    def _add_paths(self, paths):
        new = []
        for p in paths:
            if os.path.isdir(p):
                new += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(AUDIO_EXT))
            elif os.path.isfile(p) and p.lower().endswith(AUDIO_EXT):
                new.append(p)
        self.files += [f for f in new if f not in self.files]
        self._show_files()
        if paths and not new:
            self.status.configure(text="No supported audio files in the dropped items.")

    def pick_files(self):
        ps = filedialog.askopenfilenames(title="Audio files", filetypes=AUDIO_TYPES,
                                         initialdir=self.cfg.get("last_in") or None)
        if ps:
            self.cfg["last_in"] = os.path.dirname(ps[0])
            self._add_paths(ps)

    def pick_folder(self):
        d = filedialog.askdirectory(title="Folder with audio files", initialdir=self.cfg.get("last_in") or None)
        if d:
            self.cfg["last_in"] = d
            self._add_paths([d])

    def clear_files(self):
        self.files = []
        self._show_files()

    def _show_files(self):
        n = len(self.files)
        p = PALETTE[self.theme]
        if not n:
            txt = ("⤓   Drop audio files or folders here, or click to browse" if self.dnd
                   else "Click to browse for audio files")
            self.drop.configure(text=txt, foreground=p["muted"])
        else:
            names = ", ".join(os.path.basename(f) for f in self.files[:4]) + (" …" if n > 4 else "")
            self.drop.configure(text=f"♫   {n} file{'s' if n > 1 else ''} ready:  {names}", foreground=p["log_fg"])

    def pick_out(self):
        d = filedialog.askdirectory(title="Output folder", initialdir=self.out_var.get() or None)
        if d:
            self.out_var.set(d)

    def _model_changed(self):
        for w in self.stem_box.winfo_children():
            w.destroy()
        key = self.model_var.get()
        if key not in MODELS:
            key = "roformer"
            self.model_var.set(key)
        saved = self.saved_stems.get(key)
        self.stem_vars = {}
        for st in MODELS[key]["stems"]:
            v = tk.BooleanVar(value=(st in saved) if saved is not None else True)
            self.stem_vars[st] = v
            ttk.Checkbutton(self.stem_box, text=st, variable=v, command=self._remember_stems).pack(side="left",
                                                                                                padx=(0, 4))

    def _remember_stems(self):
        self.saved_stems[self.model_var.get()] = [s for s, v in self.stem_vars.items() if v.get()]

    def _save_settings(self):
        self._remember_stems()
        self.cfg.update(model=self.model_var.get(), stems=self.saved_stems, output_dir=self.out_var.get(),
                        subfolder=self.sub_var.get(), format=self.fmt_var.get(), quality=self.qual_var.get(),
                        device=self.dev_var.get(), autostart=self.auto_var.get(), theme=self.theme)
        try:
            save_json(SETTINGS_FILE, self.cfg)
        except OSError:
            pass

    def add_jobs(self):
        if not self.files:
            messagebox.showwarning(APP_NAME, "Add some audio files first (drop them or use “Add files…”).")
            return
        stems = [s for s, v in self.stem_vars.items() if v.get()]
        if not stems:
            messagebox.showwarning(APP_NAME, "Select at least one stem to save.")
            return
        out = self.out_var.get().strip() or DEFAULT_OUT
        q = self.qual_var.get()
        shifts, overlap = QUALITY.get(q, QUALITY["Normal"])
        for f in self.files:
            self.jobs.append({
                "id": uuid.uuid4().hex[:10], "input": f, "model": self.model_var.get(), "stems": stems,
                "output_dir": out, "subfolder": self.sub_var.get(), "format": self.fmt_var.get(),
                "quality": q, "shifts": shifts, "overlap": overlap,
                "device": "cpu" if self.dev_var.get() == "CPU only" else "auto",
                "status": WAITING, "info": "", "added": time.strftime("%Y-%m-%d %H:%M"),
            })
        self.status.configure(text=f"{len(self.files)} job(s) added to the queue.")
        self.files = []
        self._show_files()
        self._save_settings()
        self.persist()
        self.refresh_tree()

    # ------------------------------------------------------------ queue
    def job(self, jid):
        return next((j for j in self.jobs if j["id"] == jid), None)

    def persist(self):
        try:
            save_json(JOBS_FILE, self.jobs)
        except OSError:
            pass

    @staticmethod
    def row_values(j):
        if j.get("type") == "models":
            mode = "re-download" if j.get("mode") == "force" else "missing only"
            return (f"⟳  Model update ({mode})", ", ".join(j["models"]), "—", "—",
                    STATUS_TEXT[j["status"]], j.get("info", ""))
        return (os.path.basename(j["input"]), j["model"], ", ".join(j["stems"]), j["format"],
                STATUS_TEXT[j["status"]], j.get("info", ""))

    def refresh_tree(self):
        sel = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        for j in self.jobs:
            self.tree.insert("", "end", iid=j["id"], values=self.row_values(j), tags=(j["status"],))
        for i in sel:
            if self.tree.exists(i):
                self.tree.selection_add(i)

    def update_row(self, j):
        if self.tree.exists(j["id"]):
            self.tree.item(j["id"], values=self.row_values(j), tags=(j["status"],))

    def selected(self):
        return [self.job(i) for i in self.tree.selection() if self.job(i)]

    def retry_sel(self):
        for j in self.selected():
            if j["status"] in (DONE, FAILED, STOPPED):
                j["status"], j["info"] = WAITING, ""
        self.persist()
        self.refresh_tree()
        if self.running:
            self._next()

    def remove_sel(self):
        ids = {j["id"] for j in self.selected() if j["id"] != self.cur}
        self.jobs = [j for j in self.jobs if j["id"] not in ids]
        self.persist()
        self.refresh_tree()

    def clear_finished(self):
        self.jobs = [j for j in self.jobs if j["status"] in (WAITING, RUNNING)]
        self.persist()
        self.refresh_tree()

    def move_sel(self, d):
        sel = self.tree.selection()
        if len(sel) != 1:
            return
        i = next(k for k, j in enumerate(self.jobs) if j["id"] == sel[0])
        k = i + d
        if 0 <= k < len(self.jobs):
            self.jobs[i], self.jobs[k] = self.jobs[k], self.jobs[i]
            self.persist()
            self.refresh_tree()
            self.tree.see(sel[0])

    def open_out(self):
        sel = self.selected()
        d = (sel[0].get("result_dir") or sel[0]["output_dir"]) if sel else (self.out_var.get() or DEFAULT_OUT)
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def show_log(self):
        sel = self.tree.selection()
        lines = self.logs.get(sel[0], []) if sel else []
        if sel and not lines:
            j = self.job(sel[0])
            if j and j.get("files"):
                lines = ["Output:"] + [f"  {f}" for f in j["files"]]
        self.log_txt.configure(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.insert("end", "\n".join(lines))
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")

    def append_log(self, jid, line):
        buf = self.logs.setdefault(jid, [])
        buf.append(line)
        del buf[:-500]
        sel = self.tree.selection()
        if sel and sel[0] == jid:
            self.log_txt.configure(state="normal")
            self.log_txt.insert("end", ("\n" if len(buf) > 1 else "") + line)
            self.log_txt.see("end")
            self.log_txt.configure(state="disabled")

    # ------------------------------------------------------------ models
    def models_dialog(self):
        if getattr(self, "mdlg", None) and self.mdlg.winfo_exists():
            self.mdlg.lift()
            return
        d = self.mdlg = tk.Toplevel(self.root)
        d.title("Models")
        d.transient(self.root)
        d.resizable(False, False)
        fr = ttk.Frame(d, padding=16)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text="Models", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(fr, text="Stored in the “models” folder next to the app. Downloads and updates run as jobs "
                           "(placed before waiting jobs), so they never clash with a running separation.",
                  style="Small.TLabel", wraplength=560).pack(anchor="w", pady=(2, 10))
        box = ttk.Frame(fr, style="Card.TFrame", padding=12)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        self.mdl_vars, self.mdl_lbls = {}, {}
        for r, (key, spec) in enumerate(MODELS.items()):
            v = tk.BooleanVar(value=True)
            self.mdl_vars[key] = v
            ttk.Checkbutton(box, text=f"{spec['name']}  ({spec['title']})", variable=v).grid(
                row=r, column=0, sticky="w", pady=3)
            lb = ttk.Label(box, text="")
            lb.grid(row=r, column=1, sticky="e", padx=(16, 0))
            self.mdl_lbls[key] = lb
        bt = ttk.Frame(fr)
        bt.pack(fill="x", pady=(12, 0))
        ttk.Button(bt, text="⬇  Download missing", style="Accent.TButton",
                   command=lambda: self.queue_models("missing")).pack(side="left")
        ttk.Button(bt, text="⟳  Re-download selected",
                   command=lambda: self.queue_models("force")).pack(side="left", padx=6)
        ttk.Button(bt, text="📁  Folder", command=self.open_models).pack(side="left")
        ttk.Button(bt, text="Close", command=d.destroy).pack(side="right")
        self._refresh_models_dialog()
        d.after(30, lambda: set_dark_titlebar(d, self.theme == "dark"))

    def _refresh_models_dialog(self):
        if not (getattr(self, "mdlg", None) and self.mdlg.winfo_exists()):
            return
        p = PALETTE[self.theme]
        for key, (ok, mb, date) in worker.model_status().items():
            if ok:
                txt, col = f"✔  downloaded · {mb:.0f} MB" + (f" · {date}" if date else ""), p["ok"]
            else:
                txt, col = "✖  not downloaded", p["err"]
            self.mdl_lbls[key].configure(text=txt, foreground=col)

    def open_models(self):
        d = worker.models_dir()
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def queue_models(self, mode):
        keys = [k for k, v in self.mdl_vars.items() if v.get()]
        if mode == "missing":
            st = worker.model_status()
            keys = [k for k in keys if not st[k][0]]
            if not keys:
                messagebox.showinfo("Models", "The selected models are already downloaded.", parent=self.mdlg)
                return
        elif not keys:
            messagebox.showwarning("Models", "Select at least one model.", parent=self.mdlg)
            return
        elif not messagebox.askyesno("Re-download", f"Re-download {len(keys)} model(s) and the model list "
                                     "from the source?\nIf a download fails, the old file is kept.",
                                     parent=self.mdlg):
            return
        j = {"id": uuid.uuid4().hex[:10], "type": "models", "mode": mode, "models": keys,
             "input": "", "output_dir": worker.models_dir(), "status": WAITING, "info": "",
             "added": time.strftime("%Y-%m-%d %H:%M")}
        k = next((i for i, x in enumerate(self.jobs) if x["status"] == WAITING), len(self.jobs))
        self.jobs.insert(k, j)   # before the first waiting job: runs right after the current one
        self.persist()
        self.refresh_tree()
        self.status.configure(text="Model update queued.")
        self.start_queue()

    # ------------------------------------------------------------ running
    def _buttons(self):
        self.start_btn.configure(state="disabled" if self.running else "normal")
        self.pause_btn.configure(state="normal" if self.running else "disabled")
        self.kill_btn.configure(state="normal" if self.proc else "disabled")

    def start_queue(self):
        self.running = True
        self._buttons()
        self._next()

    def pause_queue(self):
        self.running = False
        self._buttons()
        self.status.configure(text="Paused: the current job will finish, then the queue stops."
                              if self.proc else "Queue paused.")

    def kill_current(self):
        if self.proc and self.proc.is_alive():
            if not messagebox.askyesno(APP_NAME, "Stop the running job? Partial results are discarded."):
                return
            self.proc.terminate()
            j = self.job(self.cur)
            if j:
                j["status"], j["info"] = STOPPED, "stopped by user"
                self.update_row(j)
            self._finish_current()

    def _next(self):
        if self.proc or not self.running:
            return
        j = next((j for j in self.jobs if j["status"] == WAITING), None)
        if j is None:
            self.running = False
            self._buttons()
            self.cur_lbl.configure(text="Idle")
            self.pct_lbl.configure(text="")
            self.status.configure(text="Queue finished – all jobs done.")
            return
        if j.get("type") != "models" and not os.path.isfile(j["input"]):
            j["status"], j["info"] = FAILED, "input file not found"
            self.update_row(j)
            self.persist()
            self.root.after(10, self._next)
            return
        j["status"], j["info"] = RUNNING, "starting…"
        self.update_row(j)
        self.persist()
        self.cur, self.cur_t0 = j["id"], time.time()
        self.logs[j["id"]] = []
        self.progress["value"] = 0
        self.cur_lbl.configure(text=self.row_values(j)[0])
        self.proc = mp.Process(target=worker.run_job, args=(j, self.q), daemon=True)
        self.proc.start()
        self._buttons()

    def _finish_current(self):
        self.proc = None
        self.cur = None
        self.persist()
        self._buttons()
        self.progress["value"] = 0
        self.pct_lbl.configure(text="")
        self.root.after(100, self._next)

    def _poll(self):
        try:
            while True:
                jid, kind, val = self.q.get_nowait()
                j = self.job(jid)
                if j is None or jid != self.cur:
                    continue
                if kind == "log":
                    self.append_log(jid, val)
                elif kind == "phase":
                    j["phase"] = val
                elif kind == "progress":
                    self.progress["value"] = val
                    self.pct_lbl.configure(text=f"{val:.0f}%")
                    j["pct"] = val
                elif kind == "device":
                    j["dev"] = val
                elif kind == "done":
                    j["status"] = DONE
                    j["files"], j["result_dir"] = val["files"], val["out_dir"]
                    if j.get("type") == "models":
                        j["info"] = f"{len(val['files'])} model(s) ready in {fmt_dur(val['seconds'])}"
                        self.root.after(50, self._refresh_models_dialog)
                    else:
                        j["info"] = f"{len(val['files'])} stems in {fmt_dur(val['seconds'])} · {j.get('dev', '')}"
                    self.append_log(jid, "Finished: " + val["out_dir"])
                    self.update_row(j)
                    self.proc.join(timeout=5)
                    self._finish_current()
                elif kind == "error":
                    j["status"], j["info"] = FAILED, val[:200]
                    self.update_row(j)
                    self.root.after(50, self._refresh_models_dialog)
                    self.proc.join(timeout=5)
                    self._finish_current()
                if j["status"] == RUNNING and kind in ("phase", "progress", "device"):
                    el = fmt_dur(time.time() - self.cur_t0)
                    j["info"] = f"{j.get('phase', '')} · {j.get('pct', 0):.0f}% · {el}" + \
                                (f" · {j['dev']}" if j.get("dev") else "")
                    self.update_row(j)
        except queue.Empty:
            pass
        # the child exited unexpectedly (e.g. out of memory)
        if self.proc and not self.proc.is_alive() and self.q.empty():
            j = self.job(self.cur)
            if j and j["status"] == RUNNING:
                j["status"], j["info"] = FAILED, f"process exited unexpectedly (code {self.proc.exitcode})"
                self.update_row(j)
            self._finish_current()
        n_wait = sum(j["status"] == WAITING for j in self.jobs)
        if self.proc:
            self.status.configure(text=f"Running 1  ·  Waiting {n_wait}  ·  "
                                       f"elapsed {fmt_dur(time.time() - self.cur_t0)}")
        self.root.after(200, self._poll)

    def on_close(self):
        if self.proc and self.proc.is_alive():
            if not messagebox.askyesno(APP_NAME, "A job is still running. Quit anyway?\n"
                                                 "It will be restarted on the next launch."):
                return
            self.proc.terminate()
        self._save_settings()
        self.persist()
        self.root.destroy()


def main():
    mp.freeze_support()
    try:  # crisp text on high-DPI screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    worker.setup_ffmpeg()
    root, dnd = make_root()
    App(root, dnd)
    root.mainloop()


if __name__ == "__main__":
    main()
