import { useCallback, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import * as engine from "./lib/engine";
import type { Config, FrameState, StartInfo } from "./lib/types";
import VideoView, { type PickedPoint } from "./components/VideoView";
import Pad from "./components/Pad";
import Stats from "./components/Stats";

// Landmarks a user can name when placing calibration points, in field metres.
const LANDMARKS: { name: string; x: (L: number, W: number) => number; y: (L: number, W: number) => number }[] = [
  { name: "Left corner (far)", x: () => 0, y: () => 0 },
  { name: "Left corner (near)", x: () => 0, y: (_L, W) => W },
  { name: "Halfway × far touchline", x: (L) => L / 2, y: () => 0 },
  { name: "Halfway × near touchline", x: (L) => L / 2, y: (_L, W) => W },
  { name: "Right corner (far)", x: (L) => L, y: () => 0 },
  { name: "Right corner (near)", x: (L) => L, y: (_L, W) => W },
  { name: "Left penalty area (far)", x: () => 16.5, y: (_L, W) => (W - 40.32) / 2 },
  { name: "Left penalty area (near)", x: () => 16.5, y: (_L, W) => (W + 40.32) / 2 },
  { name: "Right penalty area (far)", x: (L) => L - 16.5, y: (_L, W) => (W - 40.32) / 2 },
  { name: "Right penalty area (near)", x: (L) => L - 16.5, y: (_L, W) => (W + 40.32) / 2 },
  { name: "Left goal area (far)", x: () => 5.5, y: (_L, W) => (W - 18.32) / 2 },
  { name: "Left goal area (near)", x: () => 5.5, y: (_L, W) => (W + 18.32) / 2 },
  { name: "Right goal area (far)", x: (L) => L - 5.5, y: (_L, W) => (W - 18.32) / 2 },
  { name: "Right goal area (near)", x: (L) => L - 5.5, y: (_L, W) => (W + 18.32) / 2 },
  { name: "Centre spot", x: (L) => L / 2, y: (_L, W) => W / 2 },
  { name: "Left penalty spot", x: () => 11, y: (_L, W) => W / 2 },
  { name: "Right penalty spot", x: (L) => L - 11, y: (_L, W) => W / 2 },
];

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [source, setSource] = useState("");
  const [realtime, setRealtime] = useState(true);
  const [running, setRunning] = useState(false);
  const [info, setInfo] = useState<StartInfo | null>(null);
  const [state, setState] = useState<FrameState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(false);
  const [picking, setPicking] = useState(false);
  const [picked, setPicked] = useState<PickedPoint[]>([]);
  const [nextLandmark, setNextLandmark] = useState(0);
  const [summary, setSummary] = useState<string | null>(null);
  const latest = useRef<FrameState | null>(null);
  const raf = useRef<number | null>(null);
  // Recent states keyed by frame id. The picture and the state travel by
  // different routes, so the view looks up the state for the frame it is
  // actually showing rather than the newest one — see lib/mjpeg.ts.
  const history = useRef<Map<number, FrameState>>(new Map());

  useEffect(() => {
    engine.getConfig().then(setConfig).catch((e) => setError(String(e)));
    engine.listModels().then(setModels).catch(() => {});
  }, []);

  // States arrive faster than React should re-render; coalesce to one paint
  // per animation frame.
  const onState = useCallback((s: FrameState) => {
    latest.current = s;
    const h = history.current;
    h.set(s.frame_id, s);
    // A couple of seconds is ample; the picture is never further behind than
    // the decode queue, and an unbounded map would grow for the whole session.
    if (h.size > 120) {
      const cutoff = s.frame_id - 60;
      for (const k of h.keys()) if (k < cutoff) h.delete(k);
    }
    if (raf.current == null) {
      raf.current = requestAnimationFrame(() => {
        raf.current = null;
        setState(latest.current);
      });
    }
  }, []);

  const doStart = async () => {
    setError(null);
    setSummary(null);
    if (!source.trim()) {
      setError("Enter a file path, RTSP URL or /dev/video device.");
      return;
    }
    try {
      if (config) await engine.setConfig(config);
      history.current.clear();
      const i = await engine.start(source.trim(), realtime, onState);
      setInfo(i);
      setRunning(true);
    } catch (e) {
      setError(String(e));
    }
  };

  const doStop = async () => {
    try {
      const s = await engine.stop();
      setSummary(JSON.stringify(s, null, 1).slice(0, 600));
    } catch (e) {
      setError(String(e));
    }
    setRunning(false);
    setInfo(null);
  };

  const browse = async () => {
    const f = await open({ multiple: false, filters: [{ name: "Video", extensions: ["mp4", "mov", "mkv", "avi", "webm", "ts"] }] });
    if (typeof f === "string") setSource(f);
  };

  const onPick = (x: number, y: number) => {
    if (!config) return;
    const lm = LANDMARKS[nextLandmark];
    const L = config.pitch.length_m;
    const W = config.pitch.width_m;
    setPicked((p) => [...p, { x, y, fx: lm.x(L, W), fy: lm.y(L, W) }]);
    setNextLandmark((n) => Math.min(n + 1, LANDMARKS.length - 1));
  };

  const solve = async () => {
    try {
      await engine.calibrateFromPoints(picked.map((p) => [p.x, p.y]), picked.map((p) => [p.fx, p.fy]));
      setPicking(false);
    } catch (e) {
      setError(String(e));
    }
  };

  const clearCalibration = async () => {
    setPicked([]);
    setNextLandmark(0);
    try {
      await engine.setManualCalibration(null);
    } catch {
      /* not running */
    }
  };

  const frameW = info?.width ?? state?.width ?? 1280;
  const frameH = info?.height ?? state?.height ?? 720;
  const L = config?.pitch.length_m ?? 105;
  const W = config?.pitch.width_m ?? 68;

  return (
    <div className="app">
      <header>
        <div className="brand">Sports Analytics <span className="tag">offline · football</span></div>
        <div className="source-row">
          <input value={source} onChange={(e) => setSource(e.target.value)} placeholder="video file, rtsp://…, or /dev/video0" disabled={running} />
          <button onClick={browse} disabled={running}>Browse</button>
          <label className="chk"><input type="checkbox" checked={realtime} onChange={(e) => setRealtime(e.target.checked)} disabled={running} /> real-time</label>
          {running ? <button className="danger" onClick={doStop}>Stop</button> : <button className="primary" onClick={doStart}>Start</button>}
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <main>
        <section className="left">
          <VideoView previewUrl={info?.preview_url ?? null} state={state} history={history.current} frameW={frameW} frameH={frameH} picking={picking} picked={picked} onPick={onPick} showBoxes={showBoxes} />
          <div className="toolbar">
            <label className="chk"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> boxes</label>
            <span className="spacer" />
            {!picking ? (
              <button onClick={() => { setPicking(true); setPicked([]); setNextLandmark(0); }} disabled={!running}>Calibrate pitch by hand</button>
            ) : (
              <>
                <span className="hint">Click <b>{LANDMARKS[nextLandmark].name}</b> on the picture ({picked.length} placed)</span>
                <select value={nextLandmark} onChange={(e) => setNextLandmark(Number(e.target.value))}>
                  {LANDMARKS.map((l, i) => <option key={l.name} value={i}>{l.name}</option>)}
                </select>
                <button onClick={() => setPicked((p) => p.slice(0, -1))} disabled={!picked.length}>Undo</button>
                <button className="primary" onClick={solve} disabled={picked.length < 4}>Solve ({picked.length}/4+)</button>
                <button onClick={() => setPicking(false)}>Cancel</button>
              </>
            )}
            <button onClick={clearCalibration} disabled={!running}>Clear</button>
          </div>
        </section>

        <aside className="right">
          <Pad state={state} lengthM={L} widthM={W} />
          <Stats state={state} />
          {config && (
            <details className="settings">
              <summary>Settings</summary>
              <label>Detector
                <select value={config.detection.model.replace(/^models\//, "")} onChange={(e) => setConfig({ ...config, detection: { ...config.detection, model: `models/${e.target.value}` } })} disabled={running}>
                  {models.map((m) => <option key={m} value={m}>{m}</option>)}
                  {!models.includes(config.detection.model.replace(/^models\//, "")) && <option value={config.detection.model.replace(/^models\//, "")}>{config.detection.model}</option>}
                </select>
              </label>
              <label>Class map
                <select value={config.detection.class_map} onChange={(e) => setConfig({ ...config, detection: { ...config.detection, class_map: e.target.value } })} disabled={running}>
                  <option value="coco">COCO (person + ball)</option>
                  <option value="roboflow_football">Football (player/GK/ref/ball)</option>
                </select>
              </label>
              <label>Confidence <input type="number" step="0.05" min="0.05" max="0.95" value={config.detection.conf} onChange={(e) => setConfig({ ...config, detection: { ...config.detection, conf: Number(e.target.value) } })} disabled={running} /></label>
              <label>Pitch keypoint model
                <select value={config.pitch.keypoint_model ?? ""} onChange={(e) => setConfig({ ...config, pitch: { ...config.pitch, keypoint_model: e.target.value ? `models/${e.target.value}` : null } })} disabled={running}>
                  <option value="">none (manual only)</option>
                  {models.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              <label>Pitch size (m) <span className="inline">
                <input type="number" value={config.pitch.length_m} onChange={(e) => setConfig({ ...config, pitch: { ...config.pitch, length_m: Number(e.target.value) } })} disabled={running} /> ×
                <input type="number" value={config.pitch.width_m} onChange={(e) => setConfig({ ...config, pitch: { ...config.pitch, width_m: Number(e.target.value) } })} disabled={running} />
              </span></label>
              <label>Working width <input type="number" step="160" value={config.video.target_width} onChange={(e) => setConfig({ ...config, video: { ...config.video, target_width: Number(e.target.value) } })} disabled={running} /></label>
              <label title="0 adapts to the measured detector speed; 1 detects on every frame">Detect 1 frame in <input type="number" min="0" max="12" value={config.video.detect_every} onChange={(e) => setConfig({ ...config, video: { ...config.video, detect_every: Number(e.target.value) } })} disabled={running} /></label>
              <label className="chk"><input type="checkbox" checked={config.output.record} onChange={(e) => setConfig({ ...config, output: { ...config.output, record: e.target.checked, record_path: e.target.checked ? config.output.record_path ?? "outputs/session.mp4" : config.output.record_path } })} disabled={running} /> record annotated video</label>
              <label className="chk"><input type="checkbox" checked={!!config.output.session_log} onChange={(e) => setConfig({ ...config, output: { ...config.output, session_log: e.target.checked ? "outputs/session/session.jsonl" : null } })} disabled={running} /> write session log + analytics.json</label>
            </details>
          )}
          {summary && <pre className="summary">{summary}</pre>}
        </aside>
      </main>
    </div>
  );
}
