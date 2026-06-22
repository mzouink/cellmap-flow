// compute-worker.js — module Worker that runs the inference pipeline.
//
// The service worker can't do this itself (dynamic import() is forbidden there),
// so the page spawns this worker and bridges messages between the SW and here.
// Each message: { id, segment, tail }. Reply: { id, ok, status, contentType, body }.
import { handleCf } from "./compute.js";

self.onmessage = async (e) => {
  const { id, segment, tail } = e.data;
  try {
    const r = await handleCf(segment, tail);
    const transfer = r.body instanceof ArrayBuffer ? [r.body] : [];
    self.postMessage(
      { id, ok: true, status: r.status || 200, contentType: r.contentType, body: r.body },
      transfer
    );
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.stack) || err) });
  }
};
