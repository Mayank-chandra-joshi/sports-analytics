# Sports Analytics

Offline, real-time sports tracker. A Rust engine ingests a live video stream —
RTSP camera, capture card, webcam or file — detects and tracks players,
referees and the ball, calibrates the pitch, and produces analytics in
**metres**. A Tauri desktop app drives it.

Football first. A second sport is a new `SportProfile` (field model, classes,
tracker parameters), not a new pipeline.

**Nothing touches the network at runtime.** Models are local ONNX files with a
hash manifest; ONNX Runtime is a shared library in `runtime/`. Every
dependency, model and dataset below is free to use.

Design and rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start

```bash
git clone <this repo> && cd sports-analytics
./scripts/setup.sh
```

That links the ONNX Runtime library, exports the models, picks the right
hardware acceleration for this machine and builds the engine. Safe to
re-run — after a `git pull` it rebuilds only what changed.

```bash
./scripts/setup.sh --app          # also build the desktop app
./scripts/setup.sh --cpu          # force the CPU build
./scripts/setup.sh --skip-models  # keep the models already present
```

It does not install Rust, Node, ffmpeg or Python — those want your package
manager, so it names the command for your platform and stops:

```bash
# macOS
brew install rust node ffmpeg python@3.12
# Debian/Ubuntu
sudo apt install ffmpeg python3 && curl https://sh.rustup.rs -sSf | sh
```

Then run it:

```bash
./target/release/sa run samples/clip.mp4 --preview
```

<details>
<summary>Doing it by hand instead</summary>

```bash
pip install onnxruntime             # or onnxruntime-gpu for CUDA
./scripts/setup-runtime.sh          # links it into runtime/
pip install ultralytics onnx onnxsim
python3 scripts/export-models.py
cargo build --release -p sa-cli     # add --features coreml / cuda
./target/release/sa models          # verify hashes against the manifest
```
</details>

`--preview` prints a loopback MJPEG URL you can open in any browser.

The desktop app:

```bash
npm install
npm run tauri dev        # or: npm run tauri build
```

> The Tauri build links the whole engine and needs ~4 GB free. Close other
> heavy apps first, or use `cargo build -p sports-analytics -j 2`.

---

## Hardware acceleration

CPU is the default and always works. Build with the flag for your machine to
use its accelerator — the binary still falls back to CPU if the hardware turns
out to be absent, and **logs a warning when it does**, so "is it actually
accelerated?" is answerable from the log rather than from how it feels.

| Machine | Flag | Notes |
|---|---|---|
| Apple Silicon (M1–M5) | `--features coreml` | Neural Engine + GPU |
| NVIDIA | `--features cuda` | or `tensorrt`, which is faster and needs TensorRT installed |
| Intel CPU/iGPU | `--features openvino` | |
| Windows, any DX12 GPU | `--features directml` | |

```bash
cargo build --release -p sa-cli --features coreml
cargo build --release -p sports-analytics --features coreml   # the desktop app
```

The runtime library in `runtime/` must match: a CUDA build needs the CUDA
`libonnxruntime`, CoreML needs the macOS `libonnxruntime.dylib`. `pip install
onnxruntime` gives the CPU/CoreML build; `onnxruntime-gpu` gives the CUDA one.

### macOS (Apple Silicon)

```bash
brew install rust node ffmpeg
pip3 install onnxruntime                 # provides libonnxruntime.dylib
./scripts/setup-runtime.sh               # links it into runtime/
python3 scripts/export-models.py         # re-export the ONNX models
cargo build --release -p sa-cli --features coreml
./target/release/sa bench samples/clip.mp4 --frames 60
```

If `setup-runtime.sh` cannot find the library, pass its path:
`./scripts/setup-runtime.sh /path/to/libonnxruntime.dylib`.

Webcams are addressed by avfoundation index — `sa run 0` for the first camera,
`0:0` for video+audio — rather than by a `/dev/video*` path.

## Streaming from another machine

Run the tracker on one machine and feed it video from another — useful when
the fast machine is not the one holding the footage.

```bash
# on the machine WITH the video
./scripts/run_stream.sh                    # pick a file, pick a target
./scripts/run_stream.sh clip.mp4 192.168.1.3

# on the machine RUNNING the tracker
./target/release/sa run "udp://0.0.0.0:9000" --preview
```

`run_stream.sh` opens a file picker (GUI where there is one, a numbered list
otherwise), remembers the last target IP and directory, and loops the clip
until Ctrl-C. `PORT=9001 ./scripts/run_stream.sh` to use another port.

Either side can be started first — the receiver waits for the stream.

## CLI

```bash
sa run <source> [options]      # process to completion (file) or until Ctrl-C (live)
sa bench <source> --frames 300 # throughput and per-stage latency
sa models                      # list models with manifest/hash status
```

`<source>` is a file path, `rtsp://…`, or `/dev/video0` — detected from the string.

| Option | Effect |
|---|---|
| `--preview` | serve MJPEG on loopback and print the URL |
| `--realtime` | play a file at its native rate, like a live feed |
| `--record out.mp4` | write an annotated video |
| `--log session.jsonl` | one `FrameState` per line, plus `analytics.json` at exit |
| `--model`, `--keypoints` | override the models in `sa.toml` |

---

## How it works

```
ffmpeg decode ─► detect (YOLO11/ort) ─► track (BoT-SORT) ─► teams (kit colour)
                                                │
                    keypoints every 15 frames ──┴─► homography ─► analytics (metres)
                         (own thread)                              │
                                            FrameState ─► UI / recorder / session log
```

Each stage owns a thread; channels between them are bounded and **drop-oldest**,
so a live feed is never queued — the tracker always works on what is happening
now. Stage latency is measured and reported every frame.

| Crate | Role |
|---|---|
| `sa-core` | `Frame`, `Track`, `FrameState`, config, the `FieldModel`/`SportProfile` traits |
| `sa-ingest` | sources → `Arc<Frame>`; ffmpeg subprocess, HW decode, RTSP reconnect |
| `sa-infer` | ONNX Runtime sessions: `Detector`, `Embedder`, `PitchKeypoints`; EP selection; hash-checked manifest |
| `sa-track` | Kalman + ByteTrack two-pass association + optional ReID fusion |
| `sa-pitch` | football field model, DLT + RANSAC homography, keyframe interpolation |
| `sa-identity` | kit-colour team/referee clustering in Lab, live and refreshing |
| `sa-analytics` | distance, speed, possession, passes, offside, occupation, heatmaps, ball track |
| `sa-render` | overlays, tactical pad, JPEG encode (recorder/CLI; the app draws its own) |
| `sa-engine` | the stage graph, MJPEG server, recorder, session log |
| `sa-cli` | `sa run` / `sa bench` / `sa models` |
| `src-tauri` + `src` | desktop shell and React UI |

### Rules the code keeps

- **Metres or nothing.** Every analytics function takes field coordinates. No
  calibration ⇒ no metrics, and the pad says `NOT CALIBRATED` rather than
  drawing dots in the wrong places.
- **Reject, don't clamp.** A step implying superhuman speed is a tracking
  discontinuity, so it is dropped from distance and speed — not capped, which
  would make every player's top speed exactly the ceiling.
- **Interpolate, don't extrapolate.** The keyframe track blends between two
  solved frames; past `max_gap` it returns no calibration at all.
- **Possession is defined, not assumed.** Nearest player to a *seen* ball
  within `possession_radius_m` for `possession_min_frames`. Interpolated ball
  positions never assign possession, so a gap cannot manufacture a pass.
- **No colour is named anywhere.** Teams are whatever this clip's players wear,
  clustered in Lab; the referee is the minority kit matching neither team.

---

## Configuration

[`sa.toml`](sa.toml) — every field has a working default; the file documents
what each threshold is for. The desktop app edits the common ones live.

Two that matter most on new footage:

- `detection.class_map` — `coco` for the stock YOLO11 weights (person + ball),
  `roboflow_football` for a football-trained model (player/GK/referee/ball).
- `pitch.keypoint_model` — without it, calibration is manual only (pick four
  landmarks in the app). With it, the pitch solves itself every 15 frames.

## Models

| Model | Where it comes from | Licence |
|---|---|---|
| YOLO11n/s COCO | `scripts/export-models.py` (Ultralytics) | AGPL-3.0 — free for development and open-source use |
| Football detector | train YOLO11 on Roboflow *football-players-detection* | CC BY 4.0 dataset |
| Pitch keypoints (32) | Roboflow *football-field-detection*, YOLOv8-pose export | CC BY 4.0 dataset |
| OSNet ReID | torchreid / sportsreid weights, `--reid` | MIT |

`models/manifest.toml` lists what may be loaded, with a SHA-256 per file. A
file that does not match is refused; `sa models` checks them all.

Before shipping a closed-source binary, settle the detector licence — either an
Ultralytics Enterprise licence or an Apache-2 detector (RT-DETR). The
`Detector` trait keeps that a one-file change.

## Status

Built and tested: ingest, detection, tracking, teams, pitch maths, analytics,
rendering, recorder, session log, CLI, desktop shell and UI. 28 unit tests
covering the maths that is easy to get quietly wrong — Hungarian assignment,
Kalman convergence, DLT/RANSAC on synthetic homographies, keyframe
interpolation, team clustering, possession, offside, ball gating.

Measured on a 720p clip, CPU only (8 cores, machine under load): YOLO11n at
~166 ms/frame dominates; tracking and identity are ~0.03 ms each. A GPU
execution provider is the single biggest win available — rebuild
`sa-infer` with `--features cuda` and drop a CUDA `libonnxruntime` into
`runtime/`.

Not yet built: the single-target lock cascade (M4), fMP4/MSE preview, a second
sport profile.
