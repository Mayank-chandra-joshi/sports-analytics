//! Loopback MJPEG server for the preview. `<img src="http://127.0.0.1:PORT/stream">`
//! in the webview shows the live picture on every platform with no plugin and
//! no codec negotiation. Loopback only — nothing leaves the machine.
//!
//! One producer publishes the newest JPEG; each client gets frames at its own
//! pace and never causes the producer to block. std only.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

struct Shared {
    /// (publish counter, source frame id, jpeg)
    latest: Mutex<(u64, u64, Arc<Vec<u8>>)>,
    changed: Condvar,
    stop: AtomicBool,
    clients: AtomicU64,
    /// Set by a /snapshot request that arrived with nothing published yet.
    /// The producer treats it like a client for one frame, so a snapshot
    /// works from cold instead of returning an empty body.
    want_one: AtomicBool,
}

#[derive(Clone)]
pub struct MjpegServer {
    shared: Arc<Shared>,
    port: u16,
}

impl MjpegServer {
    /// Bind on 127.0.0.1; `port` 0 picks a free one.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let port = listener.local_addr()?.port();
        let shared = Arc::new(Shared {
            latest: Mutex::new((0, 0, Arc::new(Vec::new()))),
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
            clients: AtomicU64::new(0),
            want_one: AtomicBool::new(false),
        });
        let s2 = shared.clone();
        std::thread::Builder::new().name("sa-mjpeg".into()).spawn(move || {
            listener.set_nonblocking(true).ok();
            loop {
                if s2.stop.load(Ordering::Relaxed) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        let s3 = s2.clone();
                        std::thread::spawn(move || serve(stream, s3));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => break,
                }
            }
        })?;
        Ok(Self { shared, port })
    }

    pub fn port(&self) -> u16 {
        self.port
    }
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/stream", self.port)
    }
    pub fn clients(&self) -> u64 {
        self.shared.clients.load(Ordering::Relaxed)
    }

    /// Whether the producer should encode this frame at all: someone is
    /// watching the stream, or a snapshot is waiting for one.
    pub fn wanted(&self) -> bool {
        self.shared.clients.load(Ordering::Relaxed) > 0 || self.shared.want_one.load(Ordering::Relaxed)
    }

    pub fn publish(&self, jpeg: Vec<u8>) {
        self.publish_with_id(0, jpeg)
    }

    /// Publish a frame along with the source frame id it came from, so a
    /// client can align its own overlay to the picture it is showing.
    pub fn publish_with_id(&self, frame_id: u64, jpeg: Vec<u8>) {
        self.shared.want_one.store(false, Ordering::Relaxed);
        let mut g = self.shared.latest.lock().unwrap();
        g.0 += 1;
        g.1 = frame_id;
        g.2 = Arc::new(jpeg);
        drop(g);
        self.shared.changed.notify_all();
    }

    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.changed.notify_all();
    }
}

fn serve(mut stream: TcpStream, s: Arc<Shared>) {
    // Read and discard the request line/headers.
    let mut buf = [0u8; 2048];
    stream.set_read_timeout(Some(Duration::from_millis(500))).ok();
    let n = stream.read(&mut buf).unwrap_or(0);
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/");
    if path.starts_with("/snapshot") {
        // Ask the producer for a frame and wait briefly, so the first
        // snapshot after start is a picture rather than an empty body.
        let mut have = s.latest.lock().unwrap().0 > 0;
        if !have {
            s.want_one.store(true, Ordering::Relaxed);
            let g = s.latest.lock().unwrap();
            let (g, _) = s.changed.wait_timeout(g, Duration::from_secs(3)).unwrap();
            have = g.0 > 0;
        }
        let (_, _, jpg) = s.latest.lock().unwrap().clone();
        if !have && jpg.is_empty() {
            let _ = stream.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        }
        let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", jpg.len());
        let _ = stream.write_all(&jpg);
        return;
    }
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\nConnection: close\r\nPragma: no-cache\r\n\r\n",
    );
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    stream.set_nodelay(true).ok();
    s.clients.fetch_add(1, Ordering::Relaxed);
    let mut seen = 0u64;
    loop {
        let (frame_id, jpg) = {
            let mut g = s.latest.lock().unwrap();
            while g.0 == seen && !s.stop.load(Ordering::Relaxed) {
                let (ng, _) = s.changed.wait_timeout(g, Duration::from_millis(500)).unwrap();
                g = ng;
            }
            if s.stop.load(Ordering::Relaxed) {
                break;
            }
            seen = g.0;
            (g.1, g.2.clone())
        };
        if jpg.is_empty() {
            continue;
        }
        // `X-Frame-Id` rides with each part so a client that reads the
        // multipart stream itself can align overlays exactly. An <img> tag
        // ignores it, which is why the UI uses fetch() instead.
        if write!(stream, "--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: {frame_id}\r\nContent-Length: {}\r\n\r\n", jpg.len()).is_err()
            || stream.write_all(&jpg).is_err()
            || stream.write_all(b"\r\n").is_err()
        {
            break;
        }
    }
    s.clients.fetch_sub(1, Ordering::Relaxed);
}
