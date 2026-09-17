// Thin wrapper over the Tauri commands in src-tauri/src/lib.rs.

import { invoke, Channel } from "@tauri-apps/api/core";
import type { Config, FrameState, StartInfo } from "./types";

export async function getConfig(): Promise<Config> {
  return invoke<Config>("get_config");
}

export async function setConfig(config: Config): Promise<void> {
  await invoke("set_config", { config });
}

export async function listModels(): Promise<string[]> {
  return invoke<string[]>("list_models");
}

export async function start(source: string, realtime: boolean, onState: (s: FrameState) => void): Promise<StartInfo> {
  const ch = new Channel<FrameState>();
  ch.onmessage = onState;
  return invoke<StartInfo>("start", { source, realtime, onState: ch });
}

export async function stop(): Promise<unknown> {
  return invoke("stop");
}

export async function isRunning(): Promise<boolean> {
  return invoke<boolean>("is_running");
}

export async function setManualCalibration(h: number[][] | null): Promise<void> {
  await invoke("set_manual_calibration", { h });
}

export async function calibrateFromPoints(image: [number, number][], field: [number, number][]): Promise<number[][]> {
  return invoke<number[][]>("calibrate_from_points", { image, field });
}
