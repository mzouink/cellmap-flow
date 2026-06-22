// compute.js — the CellMap Flow inference pipeline, run in a module Worker.
//
// This is the work that used to live in the service worker. It moved here
// because service workers forbid dynamic import() (zarrita lazily import()s its
// decompression codecs, and onnxruntime-web may too) — module Workers allow it,
// and this also keeps heavy compute off the main thread. The service worker now
// just intercepts Neuroglancer's requests and delegates each one here.
//
// handleCf(segment, tail) reproduces server.py's routes:
//   tail ".zattrs"      -> OME multiscales JSON
//   tail "s0/.zattrs"   -> {} (optional per-array attrs; keeps Neuroglancer quiet)
//   tail "s0/.zarray"   -> array metadata JSON
//   tail "s0/z.y.x[.c]" -> inference chunk (raw bytes)

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.bundle.min.mjs";
import { getModel, getHandle, getBlob } from "./local-store.js";
import { openArray, readRoi } from "./zarr-reader.js";
import * as P from "./pipeline.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.wasmPaths = ORT_DIST;
// Flip on to debug op placement (WebGPU vs WASM fallback) in the worker console.
// ort.env.logLevel = "verbose";
// ort.env.debug = true;

const ARGS_KEY = "__CFLOW_ARGS__";

const sessionCache = new Map(); // session key -> Promise<InferenceSession>
const arrayCache = new Map(); // model name -> Promise<{ arr, shape }>

// Build (and cache) a session from a remote URL (meta.onnxUrl) or bytes stored
// in IndexedDB by the page (meta.onnxKey, set for a local .onnx file).
function getSession(meta) {
  const key = meta.onnxKey || meta.onnxUrl;
  if (!sessionCache.has(key)) {
    sessionCache.set(
      key,
      (async () => {
        let src = meta.onnxUrl;
        if (meta.onnxKey) {
          src = await getBlob(meta.onnxKey);
          if (!src) throw new Error(`no stored ONNX bytes for ${meta.onnxKey}`);
        }
        return ort.InferenceSession.create(src, {
          executionProviders: ["webgpu", "wasm"],
        });
      })()
    );
  }
  return sessionCache.get(key);
}

async function resolveSource(meta) {
  if (meta.source.type === "local") {
    const dirHandle = await getHandle(meta.source.handleKey);
    if (!dirHandle) throw new Error(`no stored handle ${meta.source.handleKey}`);
    return { ...meta.source, dirHandle };
  }
  return meta.source;
}

async function getInputArray(name, meta) {
  if (!arrayCache.has(name)) {
    arrayCache.set(
      name,
      (async () => {
        const source = await resolveSource(meta);
        const arr = await openArray(source);
        return { arr, shape: arr.shape };
      })()
    );
  }
  return arrayCache.get(name);
}

function parseModelSegment(segment) {
  const i = segment.indexOf(ARGS_KEY);
  if (i === -1) return { name: decodeURIComponent(segment), args: null };
  const name = decodeURIComponent(segment.slice(0, i));
  let rest = segment.slice(i + ARGS_KEY.length);
  if (rest.endsWith(ARGS_KEY)) rest = rest.slice(0, -ARGS_KEY.length);
  let args = null;
  try {
    let b64 = rest.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    args = JSON.parse(atob(b64));
  } catch (e) {
    console.warn("[compute] failed to decode CFLOW args", e);
  }
  return { name, args };
}

async function handleChunk(meta, cz, cy, cx) {
  const { arr } = await getInputArray(meta.__name, meta);
  const roi = P.chunkInputRoi(meta, cz, cy, cx);
  const input = await readRoi(arr, meta, roi); // { data: Float32Array, shape: input_size }

  P.applyNormalizers(input.data, meta.input_norm);

  const session = await getSession(meta);
  const dims = [1, 1, ...meta.input_size];
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input.data, dims) };
  const results = await session.run(feeds);
  const out = results[session.outputNames[0]];
  const outDims = Array.from(out.dims);

  let nd;
  if (outDims.length === 5) nd = { data: out.data, shape: outDims.slice(1) };
  else if (outDims.length === 4) nd = { data: out.data, shape: [1, ...outDims.slice(1)] };
  else throw new Error(`unexpected model output rank ${outDims.length}`);

  nd = P.applyPostprocessors(nd, meta.postprocess);

  let zarrNd;
  if (meta.has_channel) zarrNd = P.reorderToZarr(nd); // [C,z,y,x] -> [z,y,x,c]
  else zarrNd = { data: nd.data, shape: nd.shape.slice(1) }; // squeeze channel

  return P.encodeChunk(zarrNd, P.outputDtype(meta)); // ArrayBuffer
}

// Returns { status?, contentType, body } where body is a JSON string or ArrayBuffer.
export async function handleCf(segment, tail) {
  const { name, args } = parseModelSegment(segment);
  let meta;
  if (args && args.input_size) {
    // Self-contained meta embedded in the URL (shareable link).
    meta = { ...args, __name: name };
  } else {
    // Legacy: geometry from IndexedDB, args may override norms/post.
    const stored = await getModel(name);
    if (!stored) {
      return { status: 404, contentType: "text/plain", body: `unknown model ${name}` };
    }
    meta = { ...stored, ...(args || {}), __name: name };
  }

  if (tail === ".zattrs") {
    return { contentType: "application/json", body: JSON.stringify(P.buildZattrs(meta, name)) };
  }
  if (tail === "s0/.zattrs") {
    return { contentType: "application/json", body: "{}" };
  }
  if (tail === "s0/.zarray") {
    const { shape } = await getInputArray(name, meta);
    return { contentType: "application/json", body: JSON.stringify(P.buildZarray(meta, shape)) };
  }
  const m = tail.match(/^s0\/(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$/);
  if (m) {
    const buf = await handleChunk(meta, +m[1], +m[2], +m[3]);
    return { contentType: "application/octet-stream", body: buf };
  }
  return { status: 404, contentType: "text/plain", body: "not found" };
}
