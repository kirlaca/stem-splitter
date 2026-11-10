@echo off
REM Builds the portable app.
REM   build.bat       -> CPU build:  dist\StemSplitter\StemSplitter.exe
REM   build.bat gpu   -> CUDA build: dist\StemSplitter-GPU\StemSplitter.exe
REM                      (NVIDIA GPU + recent driver; falls back to CPU automatically)
REM The models folder, output folder, job queue and settings are kept across rebuilds.
setlocal
cd /d "%~dp0"
set VENV=.venv
set REQ=requirements.txt
set OUTNAME=StemSplitter
set ORT=onnxruntime
if /i "%1"=="gpu" (
  set VENV=.venv-gpu
  set REQ=requirements-gpu.txt
  set OUTNAME=StemSplitter-GPU
  set ORT=onnxruntime-gpu
)
set OUT=dist\%OUTNAME%
set KEEP=models output jobs.json settings.json

if not exist %VENV% py -3.10 -m venv %VENV%
%VENV%\Scripts\python.exe -m pip install --upgrade pip
%VENV%\Scripts\pip.exe install -r %REQ% || goto :error

for %%K in (%KEEP%) do if exist %OUT%\%%K move %OUT%\%%K _keep_%%K >nul

%VENV%\Scripts\pyinstaller.exe --noconfirm --clean --windowed --name StemSplitter ^
  --icon assets\icon.ico --add-data "assets\icon.png;assets" ^
  --distpath dist\_build --workpath build\%OUTNAME% ^
  --collect-all audio_separator --collect-all librosa --collect-all beartype ^
  --collect-all rotary_embedding_torch --collect-all einops --collect-all julius ^
  --collect-all diffq --collect-all ml_collections --collect-all soundfile --collect-all samplerate ^
  --collect-all resampy --collect-all onnx2torch --collect-all sv_ttk --collect-all tkinterdnd2 ^
  --copy-metadata audio-separator --copy-metadata torch --copy-metadata torchaudio ^
  --copy-metadata %ORT% --copy-metadata librosa --copy-metadata numpy --copy-metadata tqdm ^
  --copy-metadata requests --copy-metadata packaging --copy-metadata pyyaml --copy-metadata beartype ^
  stem_splitter.py || goto :error

if exist %OUT% rmdir /s /q %OUT%
move dist\_build\StemSplitter %OUT% >nul
rmdir dist\_build 2>nul

REM ffmpeg next to the app (required by audio-separator)
mkdir %OUT%\ffmpeg 2>nul
%VENV%\Scripts\python.exe -c "import imageio_ffmpeg,shutil,sys; shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), sys.argv[1])" %OUT%\ffmpeg\ffmpeg.exe || goto :error

for %%K in (%KEEP%) do if exist _keep_%%K move _keep_%%K %OUT%\%%K >nul
echo.
echo Done: %OUT%\StemSplitter.exe
goto :eof

:error
for %%K in (%KEEP%) do if exist _keep_%%K move _keep_%%K %OUT%\%%K >nul
echo.
echo BUILD FAILED
exit /b 1
