# sports-analytics — Architecture

Offline, real-time sports tracker. Desktop app (Tauri 2 + React) with a Rust
engine that ingests a live video stream, detects and tracks players, referees
and the ball, calibrates the pitch, and produces pitch-space analytics.

Football first. The engine is written so a second sport is a new *profile*
(field model, detector classes, tracker parameters), not a new pipeline.

The Python POC in `../sports-reid` is the **reference implementation**: its
algorithms, thresholds and measured failure modes carry over. This document
says what is ported, what is replaced, and why.

---

## 1. Goals and constraints

| Constraint | Consequence |
|---|---|
| **Offline** — no network at runtime, ever | All models are local ONNX files with a checked-in manifest. No auto-download, no telemetry, no CDN fonts in the UI. ONNX Runtime is bundled, not fetched at build time. |
| **Live input** — RTSP / capture card / webcam / file | Ingest is a *stream* with backpressure and a frame-drop policy, not a `for frame in video` loop. The engine must never fall behind the source; it drops, it does not queue. |
| **Fast** — the reason for Rust | Zero-copy frame handoff (`Arc<Frame>`), pre-allocated tensors, one thread per stage, GPU execution providers where present, CPU fallback that still runs. Budget in §7. |
| **Desktop UI** | Tauri webview draws overlays on a `<canvas>`; the engine sends *state*, not pixels, to the UI (§6). |
| **Multi-sport later** | `SportProfile` + `FieldModel` traits from day one (§9). |
| **Free during development** | Every dependency, model, dataset and toolchain below is free to use. The only licence question (Ultralytics AGPL, §8) is deferred to the *shipping* decision; nothing in M0–M4 needs a purchase. |

Non-goals for v1: multi-camera fusion, cloud sync, mobile.

---

## 2. Component overview

```
                          ┌──────────────────────────────────────────────────────┐
  RTSP / V4L2 / file ───► │  sa-ingest        GStreamer: demux → HW decode → RGB │
                          └───────────────┬──────────────────────────────────────┘
                                          │ Arc<Frame>  (bounded channel, drop-oldest)
                                          ▼
                          ┌──────────────────────────────────────────────────────┐
                          │  sa-engine  (orchestrator, one thread per stage)     │
                          │                                                      │
                          │   ┌────────────┐   ┌────────────┐   ┌────────────┐  │
                          │   │ detect     │──►│ track      │──►│ identity   │  │
                          │   │ YOLO11 ort │   │ BoT-SORT   │   │ teams/ReID │  │
                          │   └────────────┘   └─────┬──────┘   └─────┬──────┘  │
                          │                          │                │         │
                          │   ┌────────────┐         ▼                ▼         │
                          │   │ pitch      │──► homography ──►  analytics       │
                          │   │ kp model   │   (per frame)      (pitch metres)  │
                          │   │ every K fr │                                    │
                          │   └────────────┘                                    │
                          └───────────────┬──────────────────────────────────────┘
                                          │ FrameState (binary, ~2–10 KB/frame)
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              Tauri Channel          Recorder              Session log
              → React canvas         (annotated MP4,       (JSONL / parquet,
                overlays + pad        GStreamer encode)     analytics.json)
```

Everything left of the fork is one process, one Rust binary. The UI never
touches pixels except to *display* the decoded video (§6).

---

## 3. Crate layout

Cargo workspace at `sports-analytics/`. `src-tauri` becomes a thin shell over
the engine crates.

```
sports-analytics/
├── Cargo.toml                 # [workspace]
├── crates/
│   ├── sa-core/               # Frame, Detection, Track, PitchPoint, FrameState, SportProfile,
│   │                          # FieldModel trait, config structs (serde), error types. No I/O.
│   ├── sa-ingest/             # Source enum {Rtsp, V4l2, File, Ndi?} → Stream<Arc<Frame>>.
│   │                          # GStreamer pipelines, HW decode selection, reconnect, pts/latency.
│   ├── sa-infer/              # ort sessions behind traits: Detector, PoseEstimator, Embedder,
│   │                          # PitchKeypoints, Segmenter. Pre/post-processing, EP selection,
│   │                          # model manifest + sha256 check, tensor pools.
│   ├── sa-track/              # Multi-object tracking: BoT-SORT (+ByteTrack low-score pass),
│   │                          # Kalman, Hungarian/LAPJV, camera-motion compensation (sparse flow).
│   ├── sa-pitch/              # FieldModel for football, keypoint→homography (DLT + RANSAC),
│   │                          # KeyframeTrack interpolation, CameraTracker (LK + relocalise),
│   │                          # foot_point, on_pitch, coverage.
│   ├── sa-identity/           # Kit-colour team clustering (live), ReID gallery, optional
│   │                          # single-target lock cascade (port of sports-reid TargetTracker).
│   ├── sa-analytics/          # Metres-space only: speed/distance, possession/passes, occupation,
│   │                          # offside line, heatmaps, ball track. Ring buffers for live stats.
│   ├── sa-render/             # Overlay + tactical-pad rasterisation (for the recorder and CLI).
│   ├── sa-engine/             # Stage graph, channels, backpressure, FrameState assembly,
│   │                          # recorder, session log, metrics (fps, stage latency).
│   └── sa-cli/                # Headless runner: `sa run <source>`, `sa bench`, golden tests.
├── models/                    # *.onnx + manifest.toml (name, task, input size, sha256, licence)
├── src-tauri/                 # Tauri shell: commands, Channel streaming, custom video protocol
└── src/                       # React UI: video element + canvas overlay + pad + stats panels
```

Dependency direction is strictly downward: `sa-engine` → everything;
`sa-core` → nothing. `src-tauri` depends only on `sa-engine` and `sa-core`.

---

## 4. Data model (`sa-core`)

```rust
pub struct Frame {
    pub id: u64,             // monotonically increasing, gaps = dropped frames
    pub pts: Duration,       // source timestamp
    pub width: u32, pub height: u32,
    pub data: Arc<[u8]>,     // RGB8, full source resolution, from a buffer pool
}

pub struct Detection { pub class: Class, pub bbox: BBox, pub conf: f32 }
pub enum Class { Player, Goalkeeper, Referee, Ball, Other(u16) }

pub struct Track {
    pub id: u32, pub class: Class, pub bbox: BBox,
    pub state: TrackState,   // Confirmed | Tentative | Lost { frames }
    pub team: Option<Team>,  // A | B | Referee | Unknown
    pub pitch: Option<PitchPoint>,   // metres, only when calibrated
    pub embedding: Option<Arc<[f32]>>,
}

pub struct Calibration { pub h: [[f64; 3]; 3], pub confidence: f32, pub source: CalibSource }
//  CalibSource: Keyframe | Interpolated | Tracked | Manual | None

pub struct FrameState {          // what the UI and the log receive
    pub frame_id: u64, pub pts: Duration,
    pub tracks: Vec<Track>, pub ball: Option<BallState>,
    pub calibration: Option<Calibration>,
    pub teams: Option<TeamColours>,
    pub stats: LiveStats,        // possession %, per-team distance, fps, stage latencies
}
```

`FrameState` is serialised with `rkyv` (zero-copy) or `bincode` — never JSON —
for the Tauri channel and the session log. JSON is an *export* format only.

---

## 5. Pipeline stages

Each stage is a thread owning its model/state, connected by bounded
`crossbeam` channels. Every channel between ingest and the UI is
**drop-oldest** (capacity 2): a stage that falls behind sees the newest frame,
not a backlog. Stage latency is measured and exported in `LiveStats`.

### 5.1 Ingest (`sa-ingest`)

GStreamer via `gstreamer-rs`. One pipeline string per source type, hardware
decoder chosen at startup by probing available elements:

```
rtspsrc latency=100 ! rtph264depay ! h264parse ! {nvh264dec|vah264dec|avdec_h264}
  ! videoconvert ! video/x-raw,format=RGB ! appsink drop=true max-buffers=1
```

Why GStreamer and not `ffmpeg-next`/`opencv::videoio`: it is the only Rust
path with HW decode on NVIDIA (`nvcodec`), Intel/AMD (`va`) and Windows
(`d3d11`) behind one API, it handles RTSP reconnect and jitter, and the same
library encodes the recorder output. `retina` (pure-Rust RTSP) is the
fallback if bundling GStreamer proves painful on a target platform.

Emits `Arc<Frame>` at full resolution. A second, downscaled copy (detector
input size) is produced once here so no later stage resizes.

### 5.2 Detect (`sa-infer`)

- Model: **YOLO11** (s on GPU, n on CPU) fine-tuned for football classes
  `player / goalkeeper / referee / ball`. Baseline: Ultralytics COCO weights
  (person + sports ball), which is what `sports-reid` runs today.
- Runtime: `ort` (ONNX Runtime). Execution providers in order of preference:
  TensorRT → CUDA → OpenVINO → DirectML → CPU. Selected at startup, logged.
- Pre/post-processing (letterbox, NMS, class filter) from `usls` or a small
  in-house port of it — `usls` pulls 50+ model families; we need one. Decide
  after measuring compile time.
- Ball is a separate concern: it is 8–20 px. Two options, measured in M1:
  (a) same detector at higher input (1280), (b) a second tiny model on a
  crop around the predicted ball position. `sports-reid/src/ball.py` shows
  the gating and gap policy to port either way.

### 5.3 Track (`sa-track`)

**BoT-SORT** with ByteTrack's two-pass association, Kalman on
`(cx, cy, w, h, v…)`, and camera-motion compensation from sparse LK flow on
background pixels (the same flow `sa-pitch` uses — computed once).

Candidate crates, in order:

1. `jamtrack-rs` (MIT, git-only): ByteTrack, BoT-SORT with
   `update_with_features` for ReID embeddings, OC-SORT, BoostTrack. Closest
   to what BoT-SORT in `sports-reid` does. Risk: not on crates.io, single
   maintainer — vendor it.
2. `similari` (Apache-2): a *framework* for building trackers with SIMD
   Kalman; more work, more control.
3. In-house: ~600 lines. Kalman via `nalgebra`, assignment via `pathfinding`
   (Hungarian) or an LAPJV port.

ReID embeddings (OSNet, `sa-infer`) are computed **per track every N frames**
(N=6, `perf.reembed_interval` in the POC), not per detection per frame — the
POC measured this as the dominant cost and cached it by track id.

### 5.4 Pitch calibration (`sa-pitch`)

The POC established, with measurements, that line-based auto-fit does not
converge on broadcast follow-shots (0/40 frames) and that optical-flow
tracking of a homography fails on grass (inlier ratio 0.09). What works is a
**learned keypoint model solved every K frames, interpolated between** —
`KeyframeTrack` in `sports-reid/src/pitchkp.py` (corner-quad blending,
re-solve, 30-frame max gap, hold-only-briefly at the edges). That is ported
as-is.

- Keypoint model runs on its **own thread at its own cadence** (every K=15
  frames, ~100 ms on GPU for HRNet-w48; the Roboflow 32-keypoint YOLOv8-pose
  is 10× cheaper and less precise — ship both, choose per profile).
- Keypoints → homography: DLT + RANSAC in pure Rust (`arrsac` /
  `sample-consensus` from rust-cv, or ~150 lines in-house with `nalgebra`).
  No OpenCV dependency for this.
- Every frame gets an `H` from the track (interpolated) or `None` — never a
  stale guess. `Calibration.source` says which.
- Manual calibration (drag landmarks) stays in the UI as the fallback and
  the correction tool, exactly as in the POC's step 3; presets store `H`
  with the frame size (`sports-reid/src/presets.py`).

### 5.5 Identity (`sa-identity`)

- **Teams**: live kit-colour clustering (`LiveTeamClassifier` +
  `kit_reading` in the POC — including the nadir-view background-subtracted
  variant and the contrast/paint/bounds filters that keep goalposts out of
  the clustering). Colour, not a vision model: it is free and it works. The
  Roboflow approach (SigLIP → UMAP → k-means) is the upgrade path if colour
  fails on a kit pair.
- **Single-target lock** (optional module): the POC's `TargetTracker`
  cascade — fused ReID/colour/geometry score, face/number/hair/colour vetoes,
  crowd state, continuity, SOT bridge, re-acquire paths. Port in M4, with
  golden tests against the POC's `log.csv` on the same clips. Face and OCR
  are gated behind resolution exactly as in the POC (`min_box_h`) and are
  off by default on live streams.

### 5.6 Analytics (`sa-analytics`)

Direct port of `sports-reid/src/analytics.py` — the one rule is kept:
**every function takes metres, never pixels**; no calibration, no metrics.
Speed ceiling rejection, smoothing before differentiation, possession
defined as nearest-player-within-radius-for-K-frames, passes as possession
handovers, offside on the second-last defender. Live variants keep a ring
buffer (last 30 s) and cumulative totals.

### 5.7 Outputs (`sa-engine`)

- **UI**: `FrameState` per frame over a Tauri `Channel` (§6).
- **Recorder** (optional): `sa-render` draws overlays + pad into an RGB
  buffer, GStreamer `x264enc`/`nvh264enc` → MP4. Off by default on live.
- **Session log**: `FrameState` stream to a `.jsonl`/parquet file plus
  `analytics.json` at stop — the same schema the POC embeds into MP4
  metadata, so the two tools' outputs are comparable.

---

## 6. Engine ↔ UI

**Engine runs in-process** inside the Tauri binary (one thread pool owned by
a `tauri::State<Engine>`), not as a sidecar. It is the fastest path and the
simplest to debug. The message contract (`FrameState`) is defined so the
engine *could* become a sidecar process later without touching the UI.

Panics inside a stage are caught at the stage boundary and reported as an
engine error event; the UI stays up.

### Two channels, because video and state are different problems

1. **Video pixels → `<video>` / `<img>`**. Tauri's event system is JSON and
   is not built for this. Options, in order of preference:
   - **fMP4 over a custom URI scheme → MediaSource Extensions.** The decoded
     stream is re-muxed (not re-encoded, if the source is already H.264) into
     fragmented MP4 and fed to a `<video>` element; the webview decodes in
     hardware on every platform. Lowest CPU, ~100–200 ms latency.
   - **MJPEG over a custom URI scheme → `<img>`.** Trivial, works everywhere,
     ~30 fps at 720p; CPU cost of JPEG encode. **Use this for M0/M1**, replace
     with fMP4 in M2.
   - WebRTC (`webrtcsink`): lowest latency but WebKitGTK support on Linux is
     inconsistent (tauri #10311). Not a v1 dependency.
2. **`FrameState` → overlay canvas.** `tauri::ipc::Channel<Vec<u8>>` with
   `rkyv`/`bincode` bytes — the Tauri docs single out channels as the
   streaming primitive. The React side decodes and draws boxes, rings,
   trajectories and the tactical pad on a `<canvas>` layered over the video,
   keyed by `pts` so overlay and picture stay in sync.

Commands (Rust → `#[tauri::command]`): `open_source`, `start`, `stop`,
`set_profile`, `set_calibration_manual`, `save_preset`, `load_preset`,
`start_recording`, `stop_recording`, `export_session`.

---

## 7. Performance budget

Target: **25 fps end-to-end at 1080p** on a mid-range NVIDIA GPU; **≥10 fps
at 720p on a 4-core CPU** with YOLO11n INT8 through OpenVINO. Measured in
`sa bench` on every PR, not estimated.

| Stage | GPU (RTX-class) | CPU (4 cores) | Cadence |
|---|---|---|---|
| Decode (HW) | 1–2 ms | 8–15 ms (sw) | every frame |
| Resize/letterbox (`fast_image_resize`, SIMD) | <1 ms | 1–2 ms | every frame |
| YOLO11s @640 (TensorRT fp16) / YOLO11n INT8 | 5–8 ms | 30–50 ms | every frame |
| BoT-SORT | <1 ms | <1 ms | every frame |
| OSNet ReID 256×128, batched | ~1 ms/crop | ~15 ms/crop | per track, every 6 frames |
| Pitch keypoints (HRNet-w48 / YOLOv8-pose-kp) | 100 ms / 15 ms | 3 s / 300 ms | every 15 frames, own thread |
| Homography interp + analytics | <1 ms | <1 ms | every frame |
| `FrameState` encode + channel | <0.5 ms | <0.5 ms | every frame |

Rules that make the numbers hold:

- No per-frame heap allocation of image-sized buffers: pools in `sa-ingest`
  and `sa-infer`; `ort` sessions bound to pre-allocated input tensors.
- One flow computation per frame, shared by the tracker's camera compensation
  and the pitch tracker (the POC learned this the hard way).
- Heavy models never on the frame-critical path: pitch keypoints, ReID, and
  anything optional (segmentation) run on side threads and publish their
  latest result.
- `rayon` for per-track work (crops, histograms); no `tokio` in the engine —
  it is CPU-bound, sync threads are simpler and faster. `tokio` only in the
  Tauri layer.

---

## 8. Offline model registry

`models/manifest.toml`:

```toml
[[model]]
name     = "yolo11s-football"
task     = "detect"
file     = "yolo11s-football-640.onnx"
input    = [640, 640]
classes  = ["player", "goalkeeper", "referee", "ball"]
sha256   = "…"
licence  = "AGPL-3.0 (Ultralytics) — see docs/LICENSES.md"
```

The engine refuses to load a model whose hash does not match. `ort` is built
with `download-binaries` **off**; the ONNX Runtime shared library (with the
CUDA/TensorRT/OpenVINO providers) ships in the app bundle. First launch never
needs the network.

Models and where they come from (all exported to ONNX once, offline after):

| Model | Source | Notes |
|---|---|---|
| Player/GK/ref/ball detector | Train YOLO11 on the Roboflow *football-players-detection* dataset (CC BY 4.0); baseline COCO yolo11 | **Free to use now.** Ultralytics weights/code are AGPL-3.0, which is free for development and for open-source distribution; it only requires a decision (Enterprise licence, or swap to an Apache-2 detector such as RT-DETR) if the app ships closed-source. Keep the detector behind the `Detector` trait so that swap is a model file, not a rewrite |
| Ball (small-object) | Roboflow ball dataset, or crop-based second pass | measured in M1 |
| Pitch keypoints | Roboflow *football-field-detection* 32-kp (YOLOv8-pose) — already used by the POC; PnLCalib / "No Bells, Just Whistles" HRNet — vendored in `sports-reid/src/vendor/pnlcalib` | export HRNet with `torch.onnx.export`; the POC has the keypoint spec table |
| ReID | OSNet from `sportsreid` (SoccerNet-trained, MIT) | `torchreid` exports to ONNX; the POC's `sportsreid_core.py` picks the 256×128 input |
| Segmentation (Cut Out, optional) | yolo11n-seg | off by default on live |

---

## 9. Multi-sport abstraction

```rust
pub trait FieldModel: Send + Sync {
    fn dims(&self) -> FieldDims;                      // length, width, …
    fn segments(&self) -> &[Segment];                 // painted lines, metres
    fn landmarks(&self) -> &[(LandmarkId, Point2)];   // what the keypoint model predicts
    fn on_field(&self, p: Point2, margin_m: f32) -> bool;
}

pub struct SportProfile {
    pub name: &'static str,
    pub field: Box<dyn FieldModel>,
    pub classes: ClassMap,                 // detector class id → Class
    pub detector: ModelRef, pub keypoints: ModelRef, pub reid: Option<ModelRef>,
    pub tracker: TrackerParams,            // buffer, thresholds, max reach m/s
    pub team_rule: TeamRule,               // KitColour { n_teams: 2, officials: true } | …
    pub analytics: AnalyticsSet,           // which metrics apply
}
```

Football is the first `SportProfile`. Basketball/NFL (the crowded cases the
spec warns about) change the field model, the classes, the tracker's
`max_speed`, and swap `possession` for sport-specific rules — the stage graph
does not change.

---

## 10. What is reused from `sports-reid`, and how

| POC module | Fate |
|---|---|
| `config.yaml` thresholds and their measured rationale | Become typed `Config` structs in `sa-core` with the same names and defaults; the comments move into rustdoc |
| `pitchkp.KeyframeTrack`, `camtrack.CameraTracker` | Ported 1:1 (`sa-pitch`) |
| `analytics.py` | Ported 1:1 (`sa-analytics`) |
| `pipeline.kit_reading / player_kits / LiveTeamClassifier` | Ported (`sa-identity`) |
| `pipeline.TargetTracker` cascade | Ported in M4 (`sa-identity::lock`), gated behind golden tests |
| `ball.py` gating/gap policy | Ported (`sa-analytics::ball`) |
| `view.py` (camera elevation from box aspect), `topview.py` | Ported; the nadir detector is a `SportProfile` variant |
| Streamlit UI, point-picker component | The picker's JS (`src/components/point_picker/`) moves into the React app; Streamlit is retired |
| Python stays as | the **oracle**: `sa-cli golden` runs both on the same clip and diffs `log.csv` / `analytics.json` |

---

## 11. Open-source inventory

Runtime dependencies (Rust):

| Crate | Role | Licence | Status / risk |
|---|---|---|---|
| [`ort`](https://github.com/pykeio/ort) | ONNX Runtime bindings; CUDA/TensorRT/OpenVINO/DirectML EPs | MIT / Apache-2 | 2.0 RC, widely adopted; the de-facto Rust inference path |
| [`usls`](https://github.com/jamjamjon/usls) | YOLO v5–v26, RT-DETR, RTMPose, SAM pre/post-processing on `ort` | MIT | Active (442★). Large; use as reference or feature-gated |
| [`jamtrack-rs`](https://github.com/kadu-v/jamtrack-rs) | ByteTrack, BoT-SORT (+ReID), OC-SORT, BoostTrack in pure Rust | MIT | git-only, single maintainer → **vendor** |
| [`similari`](https://github.com/insight-platform/Similari) | Tracker-building framework, SIMD Kalman, SORT/visual-SORT | Apache-2 | Alternative to the above |
| [`mot-rs`](https://github.com/LdDl/mot-rs) | Minimal ByteTrack | MIT | Reference for an in-house port |
| [`gstreamer-rs`](https://gitlab.freedesktop.org/gstreamer/gstreamer-rs) | Ingest, HW decode, RTSP, encode | MIT/Apache-2 bindings; GStreamer LGPL-2.1 (dynamic link) | Mature; bundling on Windows/macOS needs care |
| [`retina`](https://github.com/scottlamb/retina) | Pure-Rust RTSP client | MIT/Apache-2 | Fallback if GStreamer bundling fails |
| [`ffmpeg-next`](https://github.com/zmwangx/rust-ffmpeg) / `video-rs` | Alternative ingest/encode | MIT / (FFmpeg LGPL/GPL) | Only if GStreamer is rejected |
| [`arrsac`](https://github.com/rust-cv/arrsac), `sample-consensus` (rust-cv) | RANSAC for homography, pure Rust | MIT | Small, stable |
| `nalgebra`, `ndarray`, `image`, `imageproc`, `fast_image_resize`, `rayon`, `crossbeam`, `rkyv`/`bincode`, `serde`, `tracing` | Maths, images, parallelism, serialisation, logging | MIT/Apache-2 | Standard |
| [`opencv`](https://github.com/twistedfall/opencv-rust) | Optional: LK flow, findHomography | MIT bindings; OpenCV Apache-2 | **Avoid** — 300 MB+ dependency; only if pure-Rust flow proves too slow |
| [`rerun`](https://github.com/rerun-io/rerun) | Dev-only: visualise tracks/flow/homographies while debugging | MIT/Apache-2 | Not shipped |

Reference pipelines, datasets and models (Python, used offline for training/export and as oracles):

| Project | Use |
|---|---|
| [`roboflow/sports`](https://github.com/roboflow/sports) (MIT) | Football detector + pitch-keypoint datasets, team clustering reference, radar overlay |
| [`SoccerNet/sn-gamestate`](https://github.com/soccernet/sn-gamestate) + TrackLab | End-to-end game-state reconstruction baseline; the benchmark to compare against |
| [`mguti97/PnLCalib`](https://github.com/mguti97/PnLCalib), [`No-Bells-Just-Whistles`](https://github.com/mguti97/No-Bells-Just-Whistles) | Pitch keypoint + line models (HRNet); already vendored in the POC. **Licence to confirm before shipping weights** |
| [`shallowlearn/sportsreid`](https://github.com/shallowlearn/sportsreid) (MIT) | SoccerNet-trained OSNet/ViT ReID weights; vendored in the POC |
| Ultralytics YOLO11 | Detector training/export. AGPL-3.0 — see §8 |

---

## 12. Milestones

| | Deliverable | Proves |
|---|---|---|
| **M0** | Workspace, `sa-ingest` (file + RTSP + webcam), MJPEG to the webview, fps counter | Ingest, bundling, UI plumbing, offline build |
| **M1** | Detect + track players/ball; overlays live on canvas; `sa bench` | The performance budget on real hardware, GPU and CPU |
| **M2** | Pitch keyframe calibration + manual picker + presets; tactical pad; fMP4/MSE video path | Metres are real; calibration honesty (blank pad, never a wrong one) |
| **M3** | Teams, live analytics (speed/distance/possession/occupation/offside/heatmap), recorder, session export | Feature parity with the POC's analytics on live input |
| **M4** | Single-target lock cascade port with golden tests; second `SportProfile` skeleton | The identity logic survives the port; multi-sport seam is real |

---

## 13. Decisions

### Before M1 (technical)

1. **Tracker crate vs in-house.** Vendor `jamtrack-rs`, or write BoT-SORT
   in-house (~600 lines) and own it. Recommendation: vendor, measure, decide.
2. **Video-to-UI path.** MJPEG for M0/M1 is settled; fMP4/MSE vs WebRTC for
   M2 depends on WebKitGTK behaviour on the target Linux distro.
3. **Ball strategy.** High-res single pass vs second crop pass — needs the M1
   benchmark.

### Before shipping (licensing — deferred, costs nothing now)

Development uses free tooling throughout. These only matter when a binary
is distributed to customers:

1. **Detector.** YOLO11 (AGPL-3.0) is free for development. Shipping
   closed-source means either an Ultralytics Enterprise licence or switching
   the model file to an Apache-2 detector (RT-DETR). The `Detector` trait
   keeps this a one-file change.
2. **Pitch model weights.** PnLCalib / No-Bells-Just-Whistles: confirm the
   weight licence terms before they go in a shipped `models/` bundle. The
   Roboflow 32-keypoint model (CC BY 4.0 dataset) is the fallback.
3. **GStreamer** is LGPL-2.1 — free, and fine to ship as long as it stays a
   dynamically linked library (which is how it is bundled anyway).

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| GStreamer bundling on Windows/macOS | M0 ships installers on all three; `retina` + `openh264` as the pure-Rust fallback for RTSP |
| ONNX Runtime EP libraries (CUDA/TensorRT) are large and version-locked | Ship per-platform bundles; CPU/OpenVINO build is the guaranteed path |
| HRNet keypoint model too slow on CPU | Per-profile choice of the YOLOv8-pose 32-kp model; cadence K adapts to measured latency |
| Ball detection quality on broadcast | Explicit `unknown` state (POC policy) so possession never invents a pass |
| Same-kit identity confusion in crowds | The POC's crowd/scrum state and vetoes are the answer; port them with the golden tests, not from memory |
| WebKitGTK video quirks | MJPEG path always works; keep it as a settings toggle |
