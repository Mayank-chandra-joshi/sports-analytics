#!/usr/bin/env bash
# Stream a video file to a machine running the tracker.
#
#   ./scripts/run_stream.sh                  # pick a file, pick a target
#   ./scripts/run_stream.sh clip.mp4         # that file, ask for the target
#   ./scripts/run_stream.sh clip.mp4 192.168.1.3
#   ./scripts/run_stream.sh --list           # show remembered targets
#
# Encodes to H.264 and sends MPEG-TS over UDP, which is what `sa run
# udp://0.0.0.0:PORT` expects. No server needed on either end.
#
# The receiver can be started before or after this — it waits for the stream.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/sports-analytics"
mkdir -p "$STATE"
LAST_HOST="$STATE/last-host"
LAST_DIR="$STATE/last-dir"

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true); GRN=$(tput setaf 2 2>/dev/null || true)
OFF=$(tput sgr0 2>/dev/null || true)
die() { printf "\n${RED}${BOLD}error:${OFF} %s\n\n" "$1"; exit 1; }

PORT="${PORT:-9000}"
FILE=""; HOST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --list) [ -f "$LAST_HOST" ] && cat "$LAST_HOST" || echo "(no remembered target)"; exit 0 ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) if [ -z "$FILE" ]; then FILE="$1"; else HOST="$1"; fi ;;
  esac
  shift
done

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is not installed"

# ------------------------------------------------------------ pick the file --
# A GUI picker when there is a display, a numbered menu otherwise. Both beat
# typing a path with spaces and dots in it, which is what these files have.
if [ -z "$FILE" ]; then
  START_DIR="$(cat "$LAST_DIR" 2>/dev/null || echo "$HOME/Downloads")"
  [ -d "$START_DIR" ] || START_DIR="$HOME"

  if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v zenity >/dev/null 2>&1; then
    FILE=$(zenity --file-selection --title="Video to stream" \
             --filename="$START_DIR/" \
             --file-filter="Video | *.mp4 *.mkv *.mov *.avi *.webm *.ts *.m4v" \
             --file-filter="All files | *" 2>/dev/null)
    [ -z "$FILE" ] && die "no file chosen"
  else
    # Newest first: the clip you just downloaded is usually the one you want.
    mapfile -t FILES < <(find "$START_DIR" -maxdepth 1 -type f \
      \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.webm' -o -iname '*.ts' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
    [ ${#FILES[@]} -eq 0 ] && die "no videos in $START_DIR — pass a path instead"
    printf "\n${BOLD}Videos in %s${OFF}\n\n" "$START_DIR"
    for i in "${!FILES[@]}"; do
      [ "$i" -ge 20 ] && { printf "  ${DIM}… and %d more${OFF}\n" $(( ${#FILES[@]} - 20 )); break; }
      printf "  %2d) %s\n" "$((i+1))" "$(basename "${FILES[$i]}")"
    done
    printf "\n  number: "; read -r n
    [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#FILES[@]} ] || die "not a valid choice"
    FILE="${FILES[$((n-1))]}"
  fi
fi

[ -f "$FILE" ] || die "not a file: $FILE"
dirname "$FILE" > "$LAST_DIR"

# ---------------------------------------------------------- pick the target --
if [ -z "$HOST" ]; then
  PREV="$(cat "$LAST_HOST" 2>/dev/null || true)"
  if [ -n "$PREV" ]; then
    printf "\n  target IP [${BOLD}%s${OFF}]: " "$PREV"; read -r HOST
    HOST="${HOST:-$PREV}"
  else
    printf "\n  target IP (the machine running 'sa run'): "; read -r HOST
  fi
fi
[ -n "$HOST" ] || die "no target given"
echo "$HOST" > "$LAST_HOST"

# ----------------------------------------------------------------- send it --
INFO=$(ffprobe -v error -select_streams v:0 \
       -show_entries stream=width,height,avg_frame_rate -of csv=p=0 "$FILE" 2>/dev/null)
SIZE=$(du -h "$FILE" 2>/dev/null | cut -f1)

cat <<EOF

${BOLD}Streaming${OFF}
  file    $(basename "$FILE")
  format  ${INFO:-unknown}  ($SIZE)
  to      udp://$HOST:$PORT

  On the receiving machine:
    ${BOLD}./target/release/sa run "udp://0.0.0.0:$PORT" --preview${OFF}

  ${DIM}Looping until you press Ctrl-C.${OFF}
EOF

# -re        send at the video's own rate, not as fast as the CPU allows
# -g 12      keyframe twice a second, so a receiver joining mid-stream starts
#            within ~0.5 s instead of waiting for the next default keyframe
# zerolatency  do not hold frames back for compression efficiency
# pkt_size   fits one Ethernet frame; larger fragments and drops
exec ffmpeg -hide_banner -loglevel warning \
  -re -stream_loop -1 -i "$FILE" \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 12 -an \
  -f mpegts "udp://$HOST:$PORT?pkt_size=1316"
