import type { FrameState } from "../lib/types";
import { rgb } from "../lib/types";

interface Props { state: FrameState | null }

function Row({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div className="stat-row">
      <span className="stat-a">{a}</span>
      <span className="stat-label">{label}</span>
      <span className="stat-b">{b}</span>
    </div>
  );
}

export default function Stats({ state }: Props) {
  const s = state?.stats;
  const t = state?.teams ?? null;
  const calibrated = !!state?.calibration;
  return (
    <div className="stats">
      <div className="teams-header">
        <span className="swatch" style={{ background: rgb(t?.a, "#3a7bd5") }} /> Team A
        <span className="spacer" />
        Team B <span className="swatch" style={{ background: rgb(t?.b, "#d53a3a") }} />
      </div>
      {t?.referee && (
        <div className="ref-line"><span className="swatch" style={{ background: rgb(t.referee) }} /> Referee · kit confidence {(t.confidence * 100).toFixed(0)}%</div>
      )}
      {calibrated ? (
        <>
          <Row label="possession" a={`${((s?.possession_a ?? 0) * 100).toFixed(0)}%`} b={`${((s?.possession_b ?? 0) * 100).toFixed(0)}%`} />
          <Row label="distance" a={`${((s?.distance_a_m ?? 0) / 1000).toFixed(2)} km`} b={`${((s?.distance_b_m ?? 0) / 1000).toFixed(2)} km`} />
          <Row label="passes" a={`${s?.passes_a ?? 0}`} b={`${s?.passes_b ?? 0}`} />
          <div className="stat-note">ball seen {((s?.ball_seen_rate ?? 0) * 100).toFixed(0)}% of frames{s?.offside_x != null ? ` · offside line ${s.offside_x.toFixed(1)} m` : ""}</div>
        </>
      ) : (
        <div className="stat-note">Metrics need a pitch calibration — everything here is measured in metres.</div>
      )}
      <div className="perf">
        <span>{(s?.display_fps ?? 0).toFixed(0)} fps video</span>
        <span>{(s?.fps ?? 0).toFixed(1)}/s detect</span>
        {(s?.detect_every ?? 1) > 1 && <span title="the rest are motion-predicted">1 in {s?.detect_every}</span>}
        <span>{(s?.latency.detect_ms ?? 0).toFixed(0)} ms/detect</span>
        <span>{state?.tracks.length ?? 0} tracks</span>
        {(s?.dropped ?? 0) > 0 && <span>{s?.dropped} dropped</span>}
      </div>
    </div>
  );
}
