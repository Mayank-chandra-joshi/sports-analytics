#!/usr/bin/env bash
# One-command setup: everything between `git clone` and a working binary.
#
#   ./scripts/setup.sh              # detect this machine's accelerator
#   ./scripts/setup.sh --cpu        # force the CPU build
#   ./scripts/setup.sh --app        # also build the desktop app
#   ./scripts/setup.sh --skip-models
#
# Safe to re-run: every step checks whether its work is already done, so
# after a `git pull` this just rebuilds what changed.
#
# What it does NOT do: install Rust, Node, ffmpeg or Python. Those are
# system-level and want your package manager, not a script run from a repo —
# it tells you the command instead.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true); GRN=$(tput setaf 2 2>/dev/null || true)
YLW=$(tput setaf 3 2>/dev/null || true); OFF=$(tput sgr0 2>/dev/null || true)

step() { printf "\n${BOLD}==> %s${OFF}\n" "$1"; }
ok()   { printf "    ${GRN}ok${OFF}  %s\n" "$1"; }
warn() { printf "    ${YLW}!${OFF}   %s\n" "$1"; }
die()  { printf "\n${RED}${BOLD}failed:${OFF} %s\n\n" "$1"; exit 1; }

FEATURES=""; BUILD_APP=0; SKIP_MODELS=0; FORCE_CPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) FORCE_CPU=1 ;;
    --app) BUILD_APP=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

# ---------------------------------------------------------------- 1. tools --
step "Checking prerequisites"
MISSING=""
for c in cargo ffmpeg ffprobe python3; do
  command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"
done
[ "$BUILD_APP" = 1 ] && { command -v npm >/dev/null 2>&1 || MISSING="$MISSING npm"; }

if [ -n "$MISSING" ]; then
  printf "    ${RED}missing:${OFF}%s\n\n" "$MISSING"
  case "$(uname -s)" in
    Darwin) echo "    brew install rust node ffmpeg python@3.12" ;;
    *)      echo "    sudo apt install ffmpeg python3 && curl https://sh.rustup.rs -sSf | sh" ;;
  esac
  die "install the above, then run this again"
fi
ok "cargo, ffmpeg, python3$([ "$BUILD_APP" = 1 ] && echo ', npm')"

# ------------------------------------------------- 2. hardware acceleration --
# CPU always works; a feature that the machine cannot honour costs nothing at
# runtime (ONNX Runtime falls back and the engine says so), but building the
# right one is the difference between ~140 ms and ~15 ms per frame.
step "Choosing hardware acceleration"
if [ "$FORCE_CPU" = 1 ]; then
  ok "CPU (forced with --cpu)"
elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  FEATURES="coreml"; ok "Apple Silicon detected -> coreml (Neural Engine + GPU)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  FEATURES="cuda"; ok "NVIDIA GPU detected -> cuda"
else
  ok "no supported accelerator found -> CPU"
fi

# ------------------------------------------------------------ 3. onnxruntime --
step "ONNX Runtime library"
case "$(uname -s)" in
  Darwin) LIB=libonnxruntime.dylib ;;
  MINGW*|MSYS*|CYGWIN*) LIB=onnxruntime.dll ;;
  *) LIB=libonnxruntime.so ;;
esac

if [ -e "runtime/$LIB" ]; then
  ok "already present at runtime/$LIB"
else
  if ! ./scripts/setup-runtime.sh >/dev/null 2>&1; then
    warn "not found on this machine — installing the Python wheel"
    WHEEL="onnxruntime"
    [ "$FEATURES" = "cuda" ] && WHEEL="onnxruntime-gpu"
    python3 -m pip install --quiet "$WHEEL" 2>/dev/null \
      || python3 -m pip install --quiet --break-system-packages "$WHEEL" 2>/dev/null \
      || die "could not install $WHEEL — install it yourself, then re-run"
    ./scripts/setup-runtime.sh >/dev/null 2>&1 \
      || die "installed $WHEEL but still cannot find $LIB; pass its path: ./scripts/setup-runtime.sh /path/to/$LIB"
  fi
  ok "linked runtime/$LIB"
fi

# ----------------------------------------------------------------- 4. models --
step "Models"
if [ "$SKIP_MODELS" = 1 ]; then
  ok "skipped (--skip-models)"
elif ls models/*.onnx >/dev/null 2>&1; then
  # Present but maybe never hash-recorded (e.g. copied from elsewhere, or
  # exported before local.toml existed). Cheap, and makes `sa models`
  # report them as checked rather than unverified.
  [ -f models/local.toml ] || python3 scripts/export-models.py --hashes-only >/dev/null 2>&1
  ok "already exported ($(ls models/*.onnx | wc -l | tr -d ' ') files)"
else
  python3 -c "import ultralytics" 2>/dev/null || {
    warn "installing ultralytics (one-off, for export only)"
    python3 -m pip install --quiet ultralytics onnx onnxsim 2>/dev/null \
      || python3 -m pip install --quiet --break-system-packages ultralytics onnx onnxsim 2>/dev/null \
      || die "could not install ultralytics — install it, then re-run"
  }
  echo "    exporting (takes a minute)…"
  python3 scripts/export-models.py >/dev/null 2>&1 || die "model export failed; run 'python3 scripts/export-models.py' to see why"
  ok "exported $(ls models/*.onnx 2>/dev/null | wc -l | tr -d ' ') models"
fi

# ------------------------------------------------------------------ 5. build --
step "Building the engine"
FLAG=""; [ -n "$FEATURES" ] && FLAG="--features $FEATURES"
echo "    cargo build --release -p sa-cli $FLAG"
echo "    ${DIM}(first build compiles every dependency: 5-15 minutes)${OFF}"
# shellcheck disable=SC2086
cargo build --release -p sa-cli $FLAG || die "build failed (see the error above)"
ok "target/release/sa"

if [ "$BUILD_APP" = 1 ]; then
  step "Building the desktop app"
  [ -d node_modules ] || npm install --silent || die "npm install failed"
  # shellcheck disable=SC2086
  cargo build --release -p sports-analytics $FLAG || die "app build failed"
  ok "desktop app built — run it with: npm run tauri dev${FLAG:+ -- $FLAG}"
fi

# ------------------------------------------------------------------ 6. check --
step "Verifying"
./target/release/sa models 2>/dev/null | sed 's/^/    /' || warn "could not list models"

cat <<EOF

${BOLD}Ready.${OFF}

  Track a file:     ./target/release/sa run <video.mp4> --preview
  Receive a stream: ./target/release/sa run "udp://0.0.0.0:9000" --preview
$([ "$BUILD_APP" = 1 ] && echo "  Desktop app:      npm run tauri dev${FLAG:+ -- $FLAG}")

  ${DIM}--preview prints a loopback URL you can open in a browser.${OFF}

${BOLD}Check the acceleration actually engaged.${OFF} On the first run, look for:

  ${GRN}session ready ... provider=CoreMl${OFF}     <- working
  ${YLW}WARN built with hardware acceleration
       but none is available here${OFF}          <- silently on CPU

The per-frame ${BOLD}det NNNms${OFF} figure is what governs tracking quality:
below ~40 ms every frame gets detected; above ~100 ms the tracker is
interpolating between detections and rings will lag on fast movement.
EOF
