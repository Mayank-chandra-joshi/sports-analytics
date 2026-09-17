#!/usr/bin/env bash
# Diagnose and repair the ONNX Runtime library in runtime/.
#
#   ./scripts/fix-runtime.sh          # check, and fix what it can
#   ./scripts/fix-runtime.sh --check  # report only, change nothing
#
# Run this when the engine says "failed to load onnxruntime dylib" or
# "Dlopen". It checks the things that actually break, in the order they
# break, and says which one it found rather than trying everything.
#
# Most of it is macOS-specific because that is where the failures are:
# architecture mismatches under Rosetta, and Gatekeeper quarantine.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true); GRN=$(tput setaf 2 2>/dev/null || true)
YLW=$(tput setaf 3 2>/dev/null || true); OFF=$(tput sgr0 2>/dev/null || true)
ok()    { printf "  ${GRN}ok${OFF}    %s\n" "$1"; }
bad()   { printf "  ${RED}bad${OFF}   %s\n" "$1"; }
info()  { printf "  ${DIM}·${OFF}     %s\n" "$1"; }
fixed() { printf "  ${GRN}fixed${OFF} %s\n" "$1"; }
head2() { printf "\n${BOLD}%s${OFF}\n" "$1"; }

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

case "$(uname -s)" in
  Darwin) LIB=libonnxruntime.dylib; OS=mac ;;
  MINGW*|MSYS*|CYGWIN*) LIB=onnxruntime.dll; OS=win ;;
  *) LIB=libonnxruntime.so; OS=linux ;;
esac
TARGET="runtime/$LIB"

# ------------------------------------------------------------ 1. what we have
head2 "Machine"
HOST_ARCH="$(uname -m)"
info "$(uname -s) $HOST_ARCH"
if [ "$OS" = mac ] && [ "$HOST_ARCH" = "arm64" ]; then
  # A python running under Rosetta reports x86_64 and installs an x86_64
  # wheel, whose dylib cannot load into our arm64 binary. This is the single
  # most common cause of a Dlopen failure on Apple Silicon.
  PY_ARCH="$(python3 -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
  if [ "$PY_ARCH" = "arm64" ]; then
    ok "python3 is arm64"
  else
    bad "python3 reports '$PY_ARCH' — it is running under Rosetta"
    info "its onnxruntime wheel will be x86_64 and cannot load into an arm64 binary"
    info "fix: brew install python@3.12   (then re-run this script)"
  fi
fi

# ----------------------------------------------------------- 2. does it exist
head2 "runtime/$LIB"
# `-e` follows symlinks, so a dangling link reports as absent; say which,
# because "not present" sends you looking for the wrong thing.
if [ ! -e "$TARGET" ]; then
  if [ -L "$TARGET" ]; then
    bad "symlink points at something that is not there: $(readlink "$TARGET")"
  else
    bad "not present"
  fi
  if [ "$CHECK_ONLY" = 1 ]; then exit 1; fi
  info "looking for one to link…"
  ./scripts/setup-runtime.sh >/dev/null 2>&1 && fixed "linked $TARGET" || {
    bad "none found on this machine"
    printf "\n  Install it:\n    ${BOLD}pip3 install onnxruntime${OFF}\n"
    printf "    ${DIM}(or: pip3 install --break-system-packages onnxruntime)${OFF}\n\n"
    exit 1
  }
fi

if [ -L "$TARGET" ]; then
  LINK="$(readlink "$TARGET")"
  if [ -e "$TARGET" ]; then
    ok "symlink -> $LINK"
  else
    bad "symlink is dangling -> $LINK"
    [ "$CHECK_ONLY" = 1 ] && exit 1
    rm -f "$TARGET"
    ./scripts/setup-runtime.sh >/dev/null 2>&1 && fixed "relinked" || { bad "could not relink"; exit 1; }
  fi
else
  ok "regular file ($(du -h "$TARGET" 2>/dev/null | cut -f1))"
fi

# ------------------------------------------------------------ 3. architecture
head2 "Architecture"
REAL="$(readlink -f "$TARGET" 2>/dev/null || python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$TARGET")"
if command -v file >/dev/null 2>&1; then
  DESC="$(file -b "$REAL" 2>/dev/null)"
  info "$DESC"
  case "$HOST_ARCH:$DESC" in
    arm64:*arm64*|arm64:*universal*) ok "matches this machine" ;;
    x86_64:*x86_64*|x86_64:*universal*) ok "matches this machine" ;;
    arm64:*x86_64*)
      bad "x86_64 library on an arm64 machine — this is why Dlopen fails"
      if [ "$CHECK_ONLY" = 0 ]; then
        printf "\n  ${BOLD}Fix:${OFF}\n"
        echo "    brew install python@3.12"
        echo "    /opt/homebrew/bin/pip3 install onnxruntime"
        echo "    rm runtime/$LIB && ./scripts/fix-runtime.sh"
      fi
      exit 1 ;;
    *) info "could not compare — continuing" ;;
  esac
fi

# --------------------------------------------------------------- 4. quarantine
if [ "$OS" = mac ]; then
  head2 "Gatekeeper"
  if xattr -p com.apple.quarantine "$REAL" >/dev/null 2>&1; then
    bad "quarantined — macOS will refuse to load it"
    if [ "$CHECK_ONLY" = 0 ]; then
      xattr -d com.apple.quarantine "$REAL" 2>/dev/null && fixed "quarantine removed" \
        || { bad "could not remove it (try: sudo xattr -d com.apple.quarantine '$REAL')"; exit 1; }
    fi
  else
    ok "not quarantined"
  fi
fi

# ----------------------------------------------------- 5. can it actually load
# The only test that settles it: ask the dynamic linker to load the file.
head2 "Load test"
LOADED=$(python3 - "$REAL" <<'PY' 2>&1
import ctypes, sys
try:
    ctypes.CDLL(sys.argv[1])
    print("OK")
except OSError as e:
    print(f"FAIL {e}")
PY
)
if [ "${LOADED:0:2}" = "OK" ]; then
  ok "the dynamic linker loads it"
else
  bad "${LOADED#FAIL }"
  # A symlink into a Python wheel can fail because its SIBLING libraries
  # (providers_shared etc.) are not alongside it once linked out of the
  # package. Copying the whole set is the reliable answer.
  if [ "$CHECK_ONLY" = 0 ] && [ -L "$TARGET" ]; then
    SRC_DIR="$(dirname "$REAL")"
    info "copying the library and its siblings out of $SRC_DIR"
    rm -f "$TARGET"
    cp "$SRC_DIR"/libonnxruntime*.dylib "$SRC_DIR"/libonnxruntime*.so* runtime/ 2>/dev/null
    # The engine looks for the un-versioned name.
    if [ ! -e "$TARGET" ]; then
      CAND=$(ls runtime/libonnxruntime*.dylib runtime/libonnxruntime.so* 2>/dev/null | head -1)
      [ -n "$CAND" ] && mv "$CAND" "$TARGET"
    fi
    [ "$OS" = mac ] && xattr -dr com.apple.quarantine runtime/ 2>/dev/null
    if [ -e "$TARGET" ]; then fixed "copied instead of linked — re-run to verify"; else bad "copy failed"; exit 1; fi
  else
    exit 1
  fi
fi

# --------------------------------------------------------------- 6. end to end
head2 "Engine"
if [ -x target/release/sa ]; then
  if OUT=$(./target/release/sa models 2>&1); then
    ok "the engine loads the runtime"
    echo "$OUT" | sed 's/^/        /'
  else
    bad "the engine still cannot start:"
    echo "$OUT" | head -5 | sed 's/^/        /'
    exit 1
  fi
else
  info "target/release/sa not built yet — run ./scripts/setup.sh"
fi

printf "\n${GRN}${BOLD}Runtime is working.${OFF}\n\n"
