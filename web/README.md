# CellMap Flow — Serverless (browser-side inference)

A static site that runs CellMap Flow inference **entirely in the browser**. A
service worker plays the role of [`cellmap_flow/server.py`](../cellmap_flow/server.py):
it intercepts Neuroglancer's Zarr chunk requests and computes each chunk with
[onnxruntime-web](https://onnxruntime.ai/) (WebGPU), reading the input volume
directly via [zarrita](https://github.com/manzt/zarrita.js). No backend, no GPU
server — just GitHub Pages.

This is the `serverless` branch counterpart to the Flask deployment.

## Files

| File | Role |
|---|---|
| `index.html` | UI: register a model, pick a data source, open Neuroglancer |
| `sw.js` | Service worker — fetch routing + ONNX inference (port of `server.py`) |
| `pipeline.js` | Zarr metadata, ROI math, normalizers, postprocessors, encoding |
| `zarr-reader.js` | Read an input ROI (clip + zero-pad); remote + local-FS stores |
| `local-store.js` | IndexedDB: model registry + File System Access handles |
| `vendor/neuroglancer/` | **Bundled** Neuroglancer build (you must add this) |

## How it works

The service worker intercepts **same-origin** requests and routes them:

| Same-origin URL | Backend |
|---|---|
| `…/cf/<model__CFLOW_ARGS__<b64>__CFLOW_ARGS__>/.zattrs` · `/s0/.zarray` · `/s0/z.y.x[.c]` | ONNX inference pipeline |
| `…/local/<handleKey>/<path>` | Raw Zarr file from a locally-picked directory |

Because the SW only sees same-origin traffic, **Neuroglancer must be served from
this same origin** — hence the bundled `vendor/neuroglancer/`. Remote inputs
(public S3/GCS/HTTP with CORS) are fetched by Neuroglancer directly.

### Local data — no Neuroglancer fork needed

`local://` in Neuroglancer is for annotations, not volumes. Instead the page uses
`showDirectoryPicker()` (Chromium only) to get a `FileSystemDirectoryHandle`,
stores it in IndexedDB, and the SW reads files from it, exposing the directory
behind `…/local/<handleKey>/…`. Neuroglancer sees an ordinary `zarr://` source.

## Quick start

1. **Export a model to ONNX** (one-time, needs the torch env):
   ```bash
   python scripts/export_onnx.py \
     -s models/setup16_lsd_400k_all_16_finetuned_20260427_135909.py \
     -o web/models/mito_test.onnx
   ```
   Watch the op scan for 3D `ConvTranspose` warnings (see below).

2. **Add a bundled Neuroglancer build** at `web/vendor/neuroglancer/` so it is
   served same-origin. Either:
   - copy a prebuilt `dist/` (`npm ci && npm run build-min` in a neuroglancer
     checkout, then copy `dist/min/` here), or
   - add it as a git submodule and build in CI.

3. **Serve locally** (Chrome/Edge — WebGPU + File System Access):
   ```bash
   cd web && python -m http.server 8000
   ```
   Open <http://localhost:8000>, fill the form (defaults match the fly mito test
   model), point at a CORS-enabled Zarr (or pick a local directory), and launch.

4. **Debug op placement**: uncomment `ort.env.logLevel = "verbose"` and
   `ort.env.debug = true` in `sw.js`, then watch the console to confirm ops land
   on WebGPU rather than falling back to WASM.

## ConvTranspose3D caveat

UNet decoders use 3D transposed convolutions. Conv3D landed in onnxruntime-web
1.25.0 (we pin **1.27.0**, the latest stable — 1.25.0 was never published to
npm; 1.25.1 is the first 1.25.x), but if `ConvTranspose` (3D) still falls back to WASM you
get a CPU↔GPU copy per layer. Options, in order of preference:

1. Confirm it actually falls back (verbose logging) before optimizing.
2. Re-export the model with `Resize` + `Conv3D` replacing transposed convs.
3. Accept the WASM fallback for those layers.

## v1 scope

fp32 · uncompressed chunks · single scale · `MinMax`/`Lambda`/`ZScore` norms ·
`Default`/`Threshold`/`ChannelSelection`/`Lambda` postprocessors · input voxel
size assumed equal to the array's. Out of scope (see the plan): EDT normalizer,
affinity/mutex-watershed/connected-components/Morton relabeling, Blosc chunks,
multiscale pyramids, voxel-size rescaling.

## Dependency pins

`sw.js` / `zarr-reader.js` import from jsDelivr:
`onnxruntime-web@1.27.0`, `zarrita@0.4`. If a CDN path 404s, adjust the URL
constants at the top of those files. For offline/production, vendor these into
`web/vendor/` and update the imports.
