// Mirrors `sa-core::state` (serde JSON). Keep in step with the Rust types.

export type Class = "Player" | "Goalkeeper" | "Referee" | "Ball" | { Other: number };
export type Team = "A" | "B" | "Referee" | "Unknown";
export type TrackState = "Tentative" | "Confirmed" | { Lost: { frames: number } };
export type CalibSource = "Keyframe" | "Interpolated" | "Tracked" | "Manual";

export interface BBox { x1: number; y1: number; x2: number; y2: number }
export interface Point2 { x: number; y: number }

export interface Track {
  id: number;
  class: Class;
  bbox: BBox;
  conf: number;
  state: TrackState;
  team: Team;
  pitch: Point2 | null;
  label: number;
}

export interface BallState { image: Point2; pitch: Point2 | null; seen: boolean; conf: number }

export interface Calibration { h: number[][]; confidence: number; source: CalibSource; coverage: number }

export interface TeamColours { a: [number, number, number]; b: [number, number, number]; referee: [number, number, number] | null; confidence: number }

export interface StageLatency {
  decode_ms: number; detect_ms: number; track_ms: number; identity_ms: number; pitch_ms: number; analytics_ms: number; total_ms: number;
}

export interface LiveStats {
  /** Detections per second — the rate identity is actually resolved at. */
  fps: number;
  /** Source frames per second reaching the screen. */
  display_fps: number;
  /** One detection every N source frames; the rest are motion-predicted. */
  detect_every: number;
  dropped: number; latency: StageLatency;
  possession_a: number; possession_b: number;
  distance_a_m: number; distance_b_m: number;
  passes_a: number; passes_b: number;
  ball_seen_rate: number;
  offside_x: number | null;
}

export interface FrameState {
  frame_id: number;
  pts_ms: number;
  width: number;
  height: number;
  tracks: Track[];
  ball: BallState | null;
  calibration: Calibration | null;
  teams: TeamColours | null;
  stats: LiveStats;
}

export interface StartInfo { preview_url: string | null; width: number; height: number; fps: number; frames: number }

export interface Config {
  video: { target_width: number; detect_every: number; queue_depth: number; hwaccel: boolean; loop_file: boolean };
  detection: { model: string; input_size: number; conf: number; iou_nms: number; class_map: string; threads: number };
  tracker: { high_thresh: number; low_thresh: number; match_iou: number; track_buffer: number; min_hits: number; max_speed_mps: number; appearance_weight: number };
  reid: { enabled: boolean; model: string | null; reembed_interval: number };
  pitch: { enabled: boolean; keypoint_model: string | null; keypoint_input: number; keypoint_conf: number; keyframe_every: number; max_gap: number; ransac_px: number; min_inliers: number; length_m: number; width_m: number; on_pitch_margin_m: number };
  teams: unknown;
  analytics: { enabled: boolean; possession_radius_m: number; possession_min_frames: number; speed_ceiling_mps: number; smoothing_window: number; heatmap_cells: [number, number] };
  output: { preview_quality: number; preview_width: number; draw_overlays_in_preview: boolean; record: boolean; record_path: string | null; session_log: string | null };
}

export const isLost = (s: TrackState): number | null => (typeof s === "object" && "Lost" in s ? s.Lost.frames : null);
export const isPerson = (c: Class): boolean => c === "Player" || c === "Goalkeeper" || c === "Referee";
export const rgb = (c: [number, number, number] | null | undefined, fallback = "#888"): string =>
  c ? `rgb(${c[0]},${c[1]},${c[2]})` : fallback;
