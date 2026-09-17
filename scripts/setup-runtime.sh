#!/usr/bin/env bash
# Put a libonnxruntime shared library in runtime/ so the engine can load it.
#
# The engine links ONNX Runtime dynamically (`ort` feature `load-dynamic`),
# which means one binary works with a CPU-only or a CUDA/TensorRT build —
# you swap the .so, not the executable. Resolution order at startup:
#   1. $SA_ORT_DYLIB
#   2. runtime/libonnxruntime.so next to (or above) the executable
#   3. the system loader
#
# Usage:
#   scripts/setup-runtime.sh                       # find one already on this machine
#   scripts/setup-runtime.sh /path/to/libonnxruntime.so
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime"

case "$(uname -s)" in
  Darwin) LIB=libonnxruntime.dylib ;;
  MINGW*|MSYS*|CYGWIN*) LIB=onnxruntime.dll ;;
  *) LIB=libonnxruntime.so ;;
esac

if [ $# -ge 1 ]; then
  ln -sf "$1" "$ROOT/runtime/$LIB"
  echo "linked $1 -> runtime/$LIB"
  exit 0
fi

# Anything already installed: a Python onnxruntime wheel is the usual source,
# and it is the same library the Rust side wants.
# Look where these libraries actually live, rather than scanning $HOME: a
# Python wheel's site-packages, the usual system prefixes, and any venv in
# or beside this project. Deliberately narrow — a deep scan of a home
# directory takes minutes and finds nothing extra.
SEARCH=()
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    SP=$("$py" -c 'import site,sys; print(" ".join(site.getsitepackages()+[site.getusersitepackages()]))' 2>/dev/null || true)
    for d in $SP; do [ -d "$d" ] && SEARCH+=("$d"); done
  fi
done
for d in "$ROOT/.venv" "$ROOT/../.venv" "$ROOT"/../*/.venv /usr/lib /usr/local/lib /opt/onnxruntime \
         /opt/homebrew/lib /opt/homebrew/opt/onnxruntime/lib /usr/local/opt/onnxruntime/lib; do
  [ -d "$d" ] && SEARCH+=("$d")
done

CANDIDATES=""
if [ ${#SEARCH[@]} -gt 0 ]; then
  CANDIDATES=$(find "${SEARCH[@]}" -maxdepth 6 \
    \( -name "libonnxruntime.so*" -o -name "libonnxruntime.dylib" -o -name "onnxruntime.dll" \) \
    2>/dev/null | grep -v "providers_shared" | grep -v "^$ROOT/runtime/" | head -5 || true)
fi

PICK=$(echo "$CANDIDATES" | head -1)
ln -sf "$PICK" "$ROOT/runtime/$LIB"
echo "linked $PICK -> runtime/$LIB"
echo "others found:"; echo "$CANDIDATES" | tail -n +2 | sed 's/^/  /'
