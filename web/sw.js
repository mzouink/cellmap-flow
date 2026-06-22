// sw.js — CellMap Flow serverless service worker (module worker).
//
// Plays the exact role of cellmap_flow/server.py, but client-side: intercepts
// Neuroglancer's same-origin Zarr requests and computes each chunk with
// onnxruntime-web (WebGPU), reading input data directly in the browser.
//
// Routing (all same-origin, under the SW scope):
//   .../cf/<model[__CFLOW_ARGS__<b64>__CFLOW_ARGS__]>/.zattrs        -> OME multiscales
//   .../cf/<model...>/s0/.zarray                                     -> array metadata
//   .../cf/<model...>/s0/<z>.<y>.<x>[.<c>]                           -> inference chunk
//   .../local/<handleKey>/<path...>                                  -> raw local Zarr file
//
// Register from the page with { type: "module" } so these imports work.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.bundle.min.mjs";
import { getModel, getHandle } from "./local-store.js";
import { openArray, readRoi, FileSystemStore } from "./zarr-reader.js";
import * as P from "./pipeline.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.wasmPaths = ORT_DIST;
// Flip these on while debugging op placement (WebGPU vs WASM fallback).
// ort.env.logLevel = "verbose";
// ort.env.debug = true;

const ARGS_KEY = "__CFLOW_ARGS__";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

// Per-model caches (live for the SW lifetime).
const sessionCache = new Map(); // onnxUrl -> Promise<InferenceSession>
const arrayCache = new Map(); // model name -> Promise<{ arr, shape }>

function getSession(onnxUrl) {
  if (!sessionCache.has(onnxUrl)) {
    sessionCache.set(
      onnxUrl,
      ort.InferenceSession.create(onnxUrl, {
        executionProviders: ["webgpu", "wasm"],
      })
    );
  }
  return sessionCache.get(onnxUrl);
}

async function resolveSource(meta) {
  // Attach the live FileSystemDirectoryHandle for local sources.
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

// Decode "<name>__CFLOW_ARGS__<b64>__CFLOW_ARGS__" -> { name, args }.
function parseModelSegment(segment) {
  const i = segment.indexOf(ARGS_KEY);
  if (i === -1) return { name: decodeURIComponent(segment), args: null };
  const name = decodeURIComponent(segment.slice(0, i));
  let rest = segment.slice(i + ARGS_KEY.length);
  if (rest.endsWith(ARGS_KEY)) rest = rest.slice(0, -ARGS_KEY.length);
  let args = null;
  try {
    // urlsafe base64, padding stripped (web_utils.encode_to_str).
    let b64 = rest.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    args = JSON.parse(atob(b64));
  } catch (e) {
    console.warn("[sw] failed to decode CFLOW args", e);
  }
  return { name, args };
}

const json = (obj) =>
  new Response(JSON.stringify(obj), {
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
const binary = (buf) =>
  new Response(buf, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Access-Control-Allow-Origin": "*",
    },
  });

// Merge URL-supplied norms/post (shareable links) over the stored model meta.
function effectiveMeta(meta, args) {
  if (!args) return meta;
  return {
    ...meta,
    input_norm: args.input_norm ?? meta.input_norm,
    postprocess: args.postprocess ?? meta.postprocess,
  };
}

async function handleChunk(meta, cz, cy, cx) {
  const { arr } = await getInputArray(meta.__name, meta);
  const roi = P.chunkInputRoi(meta, cz, cy, cx);
  const input = await readRoi(arr, meta, roi); // { data: Float32Array, shape: input_size }

  P.applyNormalizers(input.data, meta.input_norm);

  const session = await getSession(meta.onnxUrl);
  const dims = [1, 1, ...meta.input_size];
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input.data, dims) };
  const results = await session.run(feeds);
  const out = results[session.outputNames[0]];
  const outDims = Array.from(out.dims);

  // Normalize to channel-first [C, z, y, x].
  let nd;
  if (outDims.length === 5) nd = { data: out.data, shape: outDims.slice(1) };
  else if (outDims.length === 4) nd = { data: out.data, shape: [1, ...outDims.slice(1)] };
  else throw new Error(`unexpected model output rank ${outDims.length}`);

  nd = P.applyPostprocessors(nd, meta.postprocess);

  let zarrNd;
  if (meta.has_channel) {
    zarrNd = P.reorderToZarr(nd); // [C,z,y,x] -> [z,y,x,c]
  } else {
    zarrNd = { data: nd.data, shape: nd.shape.slice(1) }; // squeeze channel
  }
  return binary(P.encodeChunk(zarrNd, P.outputDtype(meta)));
}

// Raw passthrough for a locally-picked Zarr directory: .../local/<handleKey>/<path>
async function handleLocalFile(handleKey, path) {
  const dirHandle = await getHandle(handleKey);
  if (!dirHandle) return new Response("no handle", { status: 404 });
  const store = new FileSystemStore(dirHandle);
  const bytes = await store.get(path);
  if (bytes === undefined) return new Response("not found", { status: 404 });
  const isMeta = /\.z(array|attrs|group)$|zarr\.json$/.test(path);
  return new Response(bytes, {
    headers: {
      "Content-Type": isMeta ? "application/json" : "application/octet-stream",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

async function route(url) {
  // Local raw passthrough.
  const lidx = url.pathname.indexOf("/local/");
  if (lidx !== -1) {
    const rest = url.pathname.slice(lidx + "/local/".length).split("/");
    return handleLocalFile(rest[0], rest.slice(1).join("/"));
  }

  // Inference routes.
  const cidx = url.pathname.indexOf("/cf/");
  if (cidx === -1) return null; // not ours
  const rest = url.pathname.slice(cidx + "/cf/".length);
  const slash = rest.indexOf("/");
  const segment = slash === -1 ? rest : rest.slice(0, slash);
  const tail = slash === -1 ? "" : rest.slice(slash + 1);

  const { name, args } = parseModelSegment(segment);
  const stored = await getModel(name);
  if (!stored) return new Response(`unknown model ${name}`, { status: 404 });
  const meta = { ...effectiveMeta(stored, args), __name: name };

  if (tail === ".zattrs") return json(P.buildZattrs(meta, name));
  if (tail === "s0/.zarray") {
    const { shape } = await getInputArray(name, meta);
    return json(P.buildZarray(meta, shape));
  }
  const m = tail.match(/^s0\/(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$/);
  if (m) return handleChunk(meta, +m[1], +m[2], +m[3]);

  return new Response("not found", { status: 404 });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return; // only same-origin
  if (!url.pathname.includes("/cf/") && !url.pathname.includes("/local/")) return;
  event.respondWith(
    route(url).then((r) => r || fetch(event.request)).catch((e) => {
      console.error("[sw]", e);
      return new Response(String(e), { status: 500 });
    })
  );
});
