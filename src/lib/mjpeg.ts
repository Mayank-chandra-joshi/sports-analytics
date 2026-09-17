// Read the engine's multipart MJPEG stream ourselves, rather than pointing an
// <img> at it.
//
// WHY. The picture and the overlay reach the UI by different routes — this
// socket, and the IPC channel carrying FrameState. An <img> decodes and paints
// on the browser's own schedule, with its own buffering, so the frame on
// screen is not the frame the newest state describes. Drawing the newest state
// over it is what makes rings trail (or lead) the players.
//
// Each part carries `X-Frame-Id`, so parsing the stream ourselves lets the
// caller draw the overlay belonging to the picture it is about to show. The
// two are then in step by construction, whatever either path's latency is.

export interface MjpegFrame {
  frameId: number;
  bitmap: ImageBitmap;
}

export class MjpegReader {
  private ctrl: AbortController | null = null;
  private closed = false;

  constructor(
    private url: string,
    private onFrame: (f: MjpegFrame) => void,
    private onError?: (e: unknown) => void,
  ) {}

  /**
   * Connect and stream until `stop()`. Reconnects on a dropped connection,
   * which a long-running preview will see: the engine restarts its source,
   * the machine sleeps, the socket times out.
   */
  async start(): Promise<void> {
    this.closed = false;
    let attempt = 0;
    while (!this.closed) {
      this.ctrl = new AbortController();
      try {
        const res = await fetch(this.url, { signal: this.ctrl.signal, cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        if (!res.body) throw new Error("no response body");
        const boundary = parseBoundary(res.headers.get("content-type")) ?? "frame";
        attempt = 0;
        await this.consume(res.body.getReader(), `--${boundary}`);
      } catch (e) {
        if (this.closed) return;
        this.onError?.(e);
      }
      if (this.closed) return;
      // Back off a little so a persistently refusing endpoint does not spin.
      await new Promise((r) => setTimeout(r, Math.min(250 * 2 ** attempt++, 2000)));
    }
  }

  stop(): void {
    this.closed = true;
    this.ctrl?.abort();
    this.ctrl = null;
  }

  /** Accumulate bytes, split on the boundary, decode each JPEG part. */
  private async consume(reader: ReadableStreamDefaultReader<Uint8Array<ArrayBufferLike>>, boundary: string): Promise<void> {
    const bmark = new TextEncoder().encode(boundary);
    let buf: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
    for (;;) {
      const { done, value } = await reader.read();
      if (done || this.closed) return;
      buf = concat(buf, value);
      // Keep only the newest complete part: if we have fallen behind, the
      // older ones are stale pictures and showing them would add latency
      // rather than remove it.
      let start = indexOf(buf, bmark, 0);
      let lastPart: { headerEnd: number; bodyEnd: number } | null = null;
      while (start >= 0) {
        const next = indexOf(buf, bmark, start + bmark.length);
        if (next < 0) break;
        const headerEnd = indexOfDoubleCrlf(buf, start);
        if (headerEnd > 0 && headerEnd < next) lastPart = { headerEnd, bodyEnd: next };
        start = next;
      }
      if (!lastPart) {
        // Guard against unbounded growth on a stream that never yields a
        // second boundary (a stalled or malformed producer).
        if (buf.length > 32 * 1024 * 1024) buf = new Uint8Array(0);
        continue;
      }
      const headerText = new TextDecoder().decode(buf.subarray(0, lastPart.headerEnd));
      const frameId = parseFrameId(headerText);
      // Trailing CRLF belongs to the part, not the JPEG.
      let end = lastPart.bodyEnd;
      while (end > lastPart.headerEnd && (buf[end - 1] === 0x0a || buf[end - 1] === 0x0d)) end--;
      const jpeg = buf.slice(lastPart.headerEnd, end);
      buf = buf.slice(lastPart.bodyEnd);
      if (jpeg.length < 4) continue;
      try {
        const bitmap = await createImageBitmap(new Blob([jpeg as BlobPart], { type: "image/jpeg" }));
        if (this.closed) {
          bitmap.close();
          return;
        }
        this.onFrame({ frameId, bitmap });
      } catch {
        /* a partial or corrupt part: skip it, the next one is along shortly */
      }
    }
  }
}

function parseBoundary(contentType: string | null): string | null {
  const m = contentType?.match(/boundary=([^;]+)/i);
  return m ? m[1].trim().replace(/^"|"$/g, "") : null;
}

function parseFrameId(headers: string): number {
  const m = headers.match(/X-Frame-Id:\s*(\d+)/i);
  return m ? Number(m[1]) : 0;
}

function concat(a: Uint8Array<ArrayBufferLike>, b: Uint8Array<ArrayBufferLike>): Uint8Array<ArrayBufferLike> {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function indexOf(hay: Uint8Array<ArrayBufferLike>, needle: Uint8Array<ArrayBufferLike>, from: number): number {
  outer: for (let i = from; i <= hay.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

function indexOfDoubleCrlf(hay: Uint8Array<ArrayBufferLike>, from: number): number {
  for (let i = from; i <= hay.length - 4; i++) {
    if (hay[i] === 0x0d && hay[i + 1] === 0x0a && hay[i + 2] === 0x0d && hay[i + 3] === 0x0a) return i + 4;
  }
  return -1;
}
