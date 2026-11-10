"""StemSplitter worker: one job = one audio file (or one model update), run in its own process
so it can be cancelled at any time.

Messages go back to the GUI over a multiprocessing.Queue as (job_id, kind, value).
"""
import os
import sys
import json
import time
import shutil
import logging

# Models offered by the app. "stems" are the model's output stems (audio-separator names).
MODELS = {
    "roformer": {
        "title": "Vocals / Instrumental",
        "name": "BS-RoFormer",
        "desc": "Cleanest vocal isolation – karaoke, a cappella",
        "file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "stems": ["Vocals", "Instrumental"],
    },
    "htdemucs_ft": {
        "title": "4 stems",
        "name": "Demucs htdemucs_ft",
        "desc": "Vocals, drums, bass, other – best 4-stem quality",
        "file": "htdemucs_ft.yaml",
        "stems": ["Vocals", "Drums", "Bass", "Other"],
    },
    "htdemucs_6s": {
        "title": "6 stems",
        "name": "Demucs htdemucs_6s",
        "desc": "4 stems + guitar and piano",
        "file": "htdemucs_6s.yaml",
        "stems": ["Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"],
    },
}

ALL_STEMS = ("Vocals", "Instrumental", "Drums", "Bass", "Other", "Guitar", "Piano")
MANIFEST = "stemsplitter_models.json"


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def models_dir():
    return os.path.join(app_dir(), "models")


def setup_ffmpeg():
    """Put ffmpeg\\ffmpeg.exe next to the app on PATH (audio-separator requires ffmpeg)."""
    d = os.path.join(app_dir(), "ffmpeg")
    exe = os.path.join(d, "ffmpeg.exe")
    if not os.path.isfile(exe):
        try:  # running from source: copy the imageio-ffmpeg binary
            import imageio_ffmpeg
            os.makedirs(d, exist_ok=True)
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), exe)
        except Exception:
            pass
    if os.path.isfile(exe) and d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return os.path.isfile(exe) or shutil.which("ffmpeg") is not None


def model_status():
    """{model key: (downloaded?, MB, date)} – offline, based on the manifest / main file."""
    md = models_dir()
    try:
        with open(os.path.join(md, MANIFEST), encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        man = {}
    out = {}
    for key, spec in MODELS.items():
        files = man.get(key, {}).get("files") or [spec["file"]]
        paths = [os.path.join(md, f) for f in files]
        ok = all(os.path.isfile(p) for p in paths)
        mb = sum(os.path.getsize(p) for p in paths if os.path.isfile(p)) / 1e6
        out[key] = (ok, mb, man.get(key, {}).get("date", ""))
    return out


def _gpu_usable(torch):
    """CUDA is available AND PyTorch actually runs on the card (old cards have no kernels)."""
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA GPU / CPU-only PyTorch"
        x = torch.ones(1024, device="cuda")
        float((x * 2).sum().item())
        return True, torch.cuda.get_device_name(0)
    except Exception as e:
        return False, f"GPU not usable ({type(e).__name__}: {str(e).splitlines()[0][:120]})"


def run_job(job, q):
    """Child process entry point."""
    jid = job["id"]

    def send(kind, value=None):
        q.put((jid, kind, value))

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    try:
        if job.get("type") == "models":
            _run_models(job, send)
        else:
            _run(job, send)
    except Exception as e:
        import traceback
        send("log", traceback.format_exc())
        send("error", f"{type(e).__name__}: {e}")


def _patch_tqdm(send):
    """Redirect tqdm progress bars to the GUI (must happen BEFORE importing audio-separator)."""
    import tqdm as _tq
    state = {"bar": 0, "last": 0.0}

    class GuiTqdm(_tq.tqdm):
        def __init__(self, *a, **k):
            k["file"] = open(os.devnull, "w")
            k["mininterval"] = 0.5
            self._dl = str(k.get("unit", "")) == "iB"   # super().__init__ already calls display()
            if not self._dl:
                state["bar"] += 1
            super().__init__(*a, **k)

        def display(self, msg=None, pos=None):
            now = time.time()
            if getattr(self, "total", None) and (now - state["last"] > 0.5 or self.n >= self.total):
                state["last"] = now
                pct = min(100.0, self.n / self.total * 100)
                if self._dl:
                    if self.total < 1e6:   # small config file, not interesting
                        return True
                    send("phase", f"Downloading model {self.n / 1e6:.0f}/{self.total / 1e6:.0f} MB")
                else:
                    send("phase", f"Separating (pass {state['bar']})")
                send("progress", pct)
            return True

    _tq.tqdm = GuiTqdm
    try:
        import tqdm.auto as _tqa
        _tqa.tqdm = GuiTqdm
    except Exception:
        pass


def _select_device(job, send):
    send("phase", "Loading PyTorch...")
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    if job.get("device") == "cpu":
        use_gpu, why = False, "CPU forced"
    else:
        use_gpu, why = _gpu_usable(torch)
    if not use_gpu:
        torch.cuda.is_available = lambda: False  # makes audio-separator pick the CPU
    send("device", ("GPU: " + why) if use_gpu else "CPU")
    send("log", f"Device: {'GPU – ' + why if use_gpu else 'CPU (' + why + ')'}")


def _attach_log(send):
    class H(logging.Handler):
        def emit(self, rec):
            try:
                send("log", self.format(rec))
            except Exception:
                pass

    lg = logging.getLogger("audio_separator")
    lg.setLevel(logging.INFO)
    h = H()
    h.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(h)
    return H


def _quiet(sep, H):
    for hd in list(sep.logger.handlers):  # drop audio-separator's own stderr handler
        if isinstance(hd, logging.StreamHandler) and not isinstance(hd, H):
            sep.logger.removeHandler(hd)


def _write_manifest(key, files):
    p = os.path.join(models_dir(), MANIFEST)
    try:
        with open(p, encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        man = {}
    man[key] = {"files": files, "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)


def _model_files(sep, key):
    """Local file names belonging to a model, from the (downloaded) model list."""
    target = MODELS[key]["file"]
    for models in sep.list_supported_model_files().values():
        for info in models.values():
            if info.get("filename") == target or target in info.get("download_files", []):
                return [f.split("/")[-1] for f in info["download_files"]]
    return [target]


def _run_models(job, send):
    """Download / update models. mode: 'missing' = only missing ones, 'force' = re-download."""
    t0 = time.time()
    if not setup_ffmpeg():
        raise RuntimeError("ffmpeg not found (expected ffmpeg\\ffmpeg.exe next to the app).")
    _patch_tqdm(send)
    H = _attach_log(send)
    from audio_separator.separator import Separator
    md = models_dir()
    os.makedirs(md, exist_ok=True)
    force = job.get("mode") == "force"
    if force:  # refresh the model list from the source as well
        try:
            os.remove(os.path.join(md, "download_checks.json"))
        except OSError:
            pass
    send("phase", "Fetching model list...")
    sep = Separator(log_level=logging.INFO, model_file_dir=md, output_dir=md, info_only=True)
    _quiet(sep, H)
    done = []
    for i, key in enumerate(job["models"], 1):
        files = _model_files(sep, key)
        send("log", f"[{i}/{len(job['models'])}] {MODELS[key]['name']}: {', '.join(files)}")
        backups = []
        if force:
            for f in files:
                p = os.path.join(md, f)
                if os.path.isfile(p):
                    os.replace(p, p + ".bak")
                    backups.append(p)
        send("phase", f"Downloading {MODELS[key]['name']}...")
        try:
            sep.download_model_files(MODELS[key]["file"])
        except Exception:
            for p in backups:  # failed update: restore the old file
                if os.path.isfile(p + ".bak"):
                    os.replace(p + ".bak", p)
            raise
        for p in backups:
            try:
                os.remove(p + ".bak")
            except OSError:
                pass
        _write_manifest(key, files)
        mb = sum(os.path.getsize(os.path.join(md, f)) for f in files if os.path.isfile(os.path.join(md, f))) / 1e6
        send("log", f"   done: {mb:.0f} MB")
        done.append(key)
    send("progress", 100)
    send("done", {"files": done, "out_dir": md, "seconds": time.time() - t0})


def _run(job, send):
    t0 = time.time()
    if not setup_ffmpeg():
        raise RuntimeError("ffmpeg not found (expected ffmpeg\\ffmpeg.exe next to the app).")
    _patch_tqdm(send)
    _select_device(job, send)
    H = _attach_log(send)

    from audio_separator.separator import Separator

    spec = MODELS[job["model"]]
    src = job["input"]
    base = os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.join(job["output_dir"], base) if job.get("subfolder", True) else job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    mdir = models_dir()
    os.makedirs(mdir, exist_ok=True)

    sep = Separator(
        log_level=logging.INFO,
        model_file_dir=mdir,
        output_dir=out_dir,
        output_format=job.get("format", "WAV"),
        output_bitrate="320k" if job.get("format") == "MP3" else None,
        demucs_params={"segment_size": "Default", "shifts": int(job.get("shifts", 2)),
                       "overlap": 0.25, "segments_enabled": True},
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1,
                     "overlap": int(job.get("overlap", 8)), "pitch_shift": 0},
    )
    _quiet(sep, H)

    send("phase", "Loading model (downloads on first use)...")
    sep.load_model(model_filename=spec["file"])
    try:
        _write_manifest(job["model"], _model_files(sep, job["model"]))
    except Exception:
        pass

    prefix = (base + "_") if not job.get("subfolder", True) else ""
    names = {st: prefix + st.lower() for st in ALL_STEMS}
    if job["model"] == "roformer":  # RoFormer may name the accompaniment "Other" / "No Vocals"
        names["Other"] = names["No Vocals"] = prefix + "instrumental"

    send("phase", "Separating...")
    send("progress", 0)
    files = sep.separate(src, custom_output_names=names)

    # --- delete stems that were not requested
    wanted = {prefix + s.lower() for s in job["stems"]}
    kept = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(out_dir, f)
        if os.path.splitext(os.path.basename(p))[0] in wanted:
            kept.append(p)
        else:
            try:
                os.remove(p)
            except OSError:
                pass
    send("progress", 100)
    send("done", {"files": kept, "out_dir": out_dir, "seconds": time.time() - t0})
