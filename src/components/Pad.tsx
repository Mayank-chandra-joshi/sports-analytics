// The 2D tactical pad: a top-down pitch with every calibrated track as a
// dot in its team colour, the ball, and the offside line. Blank with an
// explicit notice when there is no calibration — never a wrong pad.

import { useEffect, useRef } from "react";
import type { FrameState } from "../lib/types";
import { isPerson, rgb } from "../lib/types";

interface Props { state: FrameState | null; lengthM: number; widthM: number }

export default function Pad({ state, lengthM, widthM }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.width;
    const H = c.height;
    const m = 12;
    const s = Math.min((W - 2 * m) / lengthM, (H - 2 * m) / widthM);
    const ox = (W - lengthM * s) / 2;
    const oy = (H - widthM * s) / 2;
    const px = (x: number, y: number): [number, number] => [ox + x * s, oy + y * s];

    ctx.fillStyle = "#2f6b2f";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.2;
    const [x0, y0] = px(0, 0);
    const [x1, y1] = px(lengthM, widthM);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.beginPath();
    ctx.moveTo(...px(lengthM / 2, 0));
    ctx.lineTo(...px(lengthM / 2, widthM));
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(...px(lengthM / 2, widthM / 2), 9.15 * s, 0, Math.PI * 2);
    ctx.stroke();
    for (const [xe, dir] of [[0, 1], [lengthM, -1]] as [number, number][]) {
      for (const [d, w] of [[16.5, 40.32], [5.5, 18.32]]) {
        const [ax, ay] = px(xe, (widthM - w) / 2);
        const [bx, by] = px(xe + dir * d, (widthM + w) / 2);
        ctx.strokeRect(Math.min(ax, bx), ay, Math.abs(bx - ax), by - ay);
      }
    }

    if (!state || !state.calibration) {
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#fff";
      ctx.font = "600 14px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("NOT CALIBRATED", W / 2, H / 2 - 4);
      ctx.font = "12px system-ui";
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillText("load a keypoint model or pick 4 landmarks", W / 2, H / 2 + 14);
      ctx.textAlign = "start";
      return;
    }
    if (state.stats.offside_x != null) {
      const [lx] = px(state.stats.offside_x, 0);
      ctx.strokeStyle = "#ffe600";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(lx, y0);
      ctx.lineTo(lx, y1);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const t of state.tracks) {
      if (!t.pitch || !isPerson(t.class)) continue;
      const col = t.team === "A" ? rgb(state.teams?.a, "#3a7bd5") : t.team === "B" ? rgb(state.teams?.b, "#d53a3a") : t.team === "Referee" ? rgb(state.teams?.referee ?? null, "#ffe600") : "#9a9a9a";
      const [x, y] = px(Math.max(-2, Math.min(lengthM + 2, t.pitch.x)), Math.max(-2, Math.min(widthM + 2, t.pitch.y)));
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    if (state.ball?.pitch) {
      const [x, y] = px(state.ball.pitch.x, state.ball.pitch.y);
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.font = "11px system-ui";
    ctx.fillText(`${state.calibration.source.toLowerCase()} · ${Math.round(state.calibration.coverage * 100)}% in shot`, m, H - 6);
  }, [state, lengthM, widthM]);

  return <canvas ref={ref} width={420} height={290} className="pad" />;
}
