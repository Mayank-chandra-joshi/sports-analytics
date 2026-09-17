// The picture (MJPEG from the engine's loopback server) with a canvas of
// overlays drawn from the latest FrameState. Also the manual-calibration
// picker: click four+ points, name their field coordinates, solve.

import { useEffect, useRef, useState } from "react";
import type { FrameState } from "../lib/types";
import { isLost, isPerson, rgb } from "../lib/types";
import { MjpegReader } from "../lib/mjpeg";

export interface PickedPoint { x: number; y: number; fx: number; fy: number }

interface Props {
  previewUrl: string | null;
  /** Reported when the stream cannot be read, so the UI can fall back. */
  onStreamError?: (e: unknown) => void;
  /** Newest state. Kept in a ref so the draw loop reads it without re-running. */
  state: FrameState | null;
  /** States by frame id, so the overlay can match the picture being shown. */
  history: Map<number, FrameState>;
  frameW: number;
  frameH: number;
  picking: boolean;
  picked: PickedPoint[];
  onPick: (x: number, y: number) => void;
  showBoxes: boolean;
}

function teamColour(team: string, teams: FrameState["teams"]): string {
  if (team === "A") return rgb(teams?.a, "#3a7bd5");
  if (team === "B") return rgb(teams?.b, "#d53a3a");
  if (team === "Referee") return rgb(teams?.referee ?? null, "#ffe600");
  return "#9a9a9a";
}

export default function VideoView({ previewUrl, onStreamError, state, history, frameW, frameH, picking, picked, onPick, showBoxes }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLCanvasElement>(null);
  // THE PICTURE, two ways.
  //
  // `img` is the default and always works: the webview's own MJPEG support,
  // no JavaScript in the path. Its drawback is that the browser decodes and
  // paints on its own schedule, so the overlay — which arrives by a separate
  // IPC channel — can be a frame or two out of step with it.
  //
  // `synced` reads the multipart stream in JS and paints each picture
  // together with the state for THAT frame id, which removes the skew. It
  // needs fetch + ReadableStream + createImageBitmap to behave on a
  // multipart response, and WebKitGTK does not always oblige.
  //
  // So: start on `img`, try `synced` alongside it, and switch only once a
  // frame has actually been painted. A capability that cannot be relied on
  // must be proven before it is depended on — the alternative is what the
  // user saw, a black rectangle where the video should be.
  const [mode, setMode] = useState<"img" | "synced">("img");
  // Set by the first painted frame; the watchdog below reads it to decide
  // whether the synced path is working at all.
  const gotFrame = useRef(false);
  // The draw loop reads these through refs, so a new state does not tear down
  // and rebuild the MJPEG connection.
  const stateRef = useRef(state);
  const histRef = useRef(history);
  const pickedRef = useRef(picked);
  const boxesRef = useRef(showBoxes);
  stateRef.current = state;
  histRef.current = history;
  pickedRef.current = picked;
  boxesRef.current = showBoxes;

  // THE PICTURE. Read the stream ourselves and paint each frame to a canvas
  // together with the overlay for THAT frame — see src/lib/mjpeg.ts for why
  // an <img> cannot keep the two in step.
  useEffect(() => {
    if (!previewUrl) return;
    // `cancelled` rather than relying on the reader alone: React 19 mounts
    // effects twice in development, and without this the second reader's
    // frames can be painted by the first's already-torn-down closure.
    let cancelled = false;
    const reader = new MjpegReader(previewUrl, ({ frameId, bitmap }) => {
      if (cancelled) {
        bitmap.close();
        return;
      }
      const vc = videoRef.current;
      if (!vc) {
        bitmap.close();
        return;
      }
      if (!gotFrame.current) {
        gotFrame.current = true;
        // Proven: the synced path is delivering pictures, so show it.
        setMode("synced");
      }
      if (vc.width !== bitmap.width || vc.height !== bitmap.height) {
        vc.width = bitmap.width;
        vc.height = bitmap.height;
      }
      vc.getContext("2d")?.drawImage(bitmap, 0, 0);
      bitmap.close();
      // Prefer the state for this exact frame; fall back to the newest one
      // when it has not arrived (the pipeline runs behind the decoder).
      const st = histRef.current.get(frameId) ?? stateRef.current;
      drawOverlay(canvasRef.current, st, frameW, frameH, pickedRef.current, boxesRef.current);
    }, (e) => {
      if (cancelled) return;
      // Not fatal: the <img> keeps showing the video. Report it so a
      // developer can see why the overlay is not frame-synced.
      console.warn("mjpeg synced path unavailable, using <img>:", e);
      onStreamError?.(e);
    });
    void reader.start();
    return () => {
      cancelled = true;
      reader.stop();
    };
  }, [previewUrl, frameW, frameH, onStreamError]);

  // A new source starts unproven again.
  useEffect(() => {
    gotFrame.current = false;
    setMode("img");
  }, [previewUrl]);

  // Redraw when the picks change even if no new frame has arrived, so the
  // calibration dots appear the moment they are placed.
  useEffect(() => {
    drawOverlay(canvasRef.current, stateRef.current, frameW, frameH, picked, showBoxes);
  }, [picked, showBoxes, frameW, frameH, state]);

  const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!picking) return;
    const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * frameW;
    const y = ((e.clientY - r.top) / r.height) * frameH;
    onPick(x, y);
  };

  return (
    <div className={`video ${picking ? "picking" : ""}`} style={{ aspectRatio: `${frameW} / ${frameH}` }} onClick={onClick}>
      {!previewUrl && <div className="video-empty">No source</div>}
      {/* Both are mounted whenever there is a source: the canvas has to
          exist for the reader to paint into and prove itself, and the <img>
          has to keep showing the video until it does. Only one is visible. */}
      {previewUrl && (
        <>
          <img src={previewUrl} alt="live" draggable={false} style={{ visibility: mode === "img" ? "visible" : "hidden" }} />
          <canvas ref={videoRef} style={{ visibility: mode === "synced" ? "visible" : "hidden" }} />
        </>
      )}
      <canvas ref={canvasRef} />
    </div>
  );
}

/** Draw tracks, the ball and calibration picks for one state. */
function drawOverlay(
  c: HTMLCanvasElement | null,
  state: FrameState | null,
  frameW: number,
  frameH: number,
  picked: PickedPoint[],
  showBoxes: boolean,
) {
  {
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    if (c.width !== frameW || c.height !== frameH) {
      c.width = frameW;
      c.height = frameH;
    }
    ctx.clearRect(0, 0, c.width, c.height);
    if (state) {
      for (const t of state.tracks) {
        const lost = isLost(t.state);
        // Matches sa-render: at a few detections per second, even one missed
        // detection puts the predicted ring beside the player rather than
        // under him. Show nothing instead — see the note there.
        if (lost !== null) continue;
        const b = t.bbox;
        const col = teamColour(t.team, state.teams);
        const cx = (b.x1 + b.x2) / 2;
        const w = Math.max(6, Math.min(60, (b.x2 - b.x1) * 0.55));
        if (isPerson(t.class)) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = col;
          ctx.fillStyle = col;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.ellipse(cx, b.y2, w, w * 0.35, 0, 0, Math.PI * 2);
          ctx.globalAlpha = 0.28;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.stroke();
          if (showBoxes) {
            ctx.lineWidth = 1;
            ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
          }
          ctx.globalAlpha = 1;
          ctx.font = "bold 13px system-ui, sans-serif";
          ctx.fillStyle = "#fff";
          ctx.strokeStyle = "rgba(0,0,0,0.7)";
          ctx.lineWidth = 3;
          const label = `#${t.label}`;
          ctx.strokeText(label, cx - 8, b.y2 + w * 0.35 + 14);
          ctx.fillText(label, cx - 8, b.y2 + w * 0.35 + 14);
        } else {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        }
      }
      if (state.ball) {
        ctx.strokeStyle = state.ball.seen ? "#fff" : "#ffa500";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(state.ball.image.x, state.ball.image.y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    // Calibration picks.
    picked.forEach((p, i) => {
      ctx.fillStyle = "#ff3b30";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px system-ui";
      ctx.fillText(`${i + 1}`, p.x + 7, p.y - 7);
    });
  }
}
