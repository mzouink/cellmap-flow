// zarr-reader.js — read an input ROI (in nm) from a Zarr array in the browser.
//
// Replaces cellmap_flow/utils/ds.py::to_ndarray_tensorstore for v1: clip the
// requested ROI to the array domain and zero-pad the out-of-bounds remainder.
// Assumes the dataset origin is 0 and its voxel size equals meta.input_voxel_size
// (no rescaling) — see plan "v1 can assume input voxel size == array voxel size".
//
// zarrita is loaded from a CDN ESM build; pin the version in web/README.md.
import * as zarr from "https://cdn.jsdelivr.net/npm/zarrita@0.4/+esm";
import { FileSystemStore } from "./local-store.js";

// Open a Zarr array from a source descriptor:
//   { type: "remote", url }            -> FetchStore
//   { type: "local",  dirHandle, path } -> FileSystemStore (path = subdir of root)
export async function openArray(source) {
  let store;
  if (source.type === "local") {
    let dir = source.dirHandle;
    for (const part of (source.path || "").split("/").filter(Boolean)) {
      dir = await dir.getDirectoryHandle(part);
    }
    store = new FileSystemStore(dir);
  } else {
    store = new zarr.FetchStore(source.url);
  }
  return zarr.open(store, { kind: "array" });
}

// Read the input ROI (nm) into a Float32Array of shape meta.input_size ([z,y,x]),
// clipping to the array and zero-padding outside.
export async function readRoi(arr, meta, inputRoiNm) {
  const ivs = meta.input_voxel_size;
  const want = meta.input_size; // [z, y, x] voxels
  // Requested voxel window (may extend past the array edges).
  const begin = inputRoiNm.begin.map((v, i) => Math.round(v / ivs[i]));

  const out = new Float32Array(want[0] * want[1] * want[2]);

  // Intersection of [begin, begin+want) with [0, arr.shape).
  const dim = arr.shape; // assume 3D (z, y, x)
  const lo = begin.map((b) => Math.max(b, 0));
  const hi = begin.map((b, i) => Math.min(b + want[i], dim[i]));
  if (hi.some((h, i) => h <= lo[i])) return { data: out, shape: [...want] }; // fully outside

  const sel = [
    zarr.slice(lo[0], hi[0]),
    zarr.slice(lo[1], hi[1]),
    zarr.slice(lo[2], hi[2]),
  ];
  const chunk = await zarr.get(arr, sel); // { data, shape, stride } C-order, native dtype
  const [cz, cy, cx] = chunk.shape;
  const src = chunk.data;

  // Destination offset of the valid block within the padded output window.
  const off = lo.map((l, i) => l - begin[i]);
  const [wz, wy, wx] = want;
  for (let z = 0; z < cz; z++) {
    for (let y = 0; y < cy; y++) {
      const srcRow = (z * cy + y) * cx;
      const dstRow = (((z + off[0]) * wy + (y + off[1])) * wx) + off[2];
      for (let x = 0; x < cx; x++) out[dstRow + x] = src[srcRow + x];
    }
  }
  return { data: out, shape: [...want] };
}
