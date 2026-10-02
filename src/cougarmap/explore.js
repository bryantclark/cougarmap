// The interactive page's kernel: the camera-spot score and the spot picking, ported from analyze.py so the page
// can rerun them in the browser when a weight changes. Pure functions over typed arrays (row-major, H x W);
// the tests run this file under node against the Python model. Keep it in step with:
//   score()  <- analyze.combine (spot, stacking ramps, habitat x season, the camera-zone Gaussian, scale/clip)
//   final()  <- analyze.apply_masks (site penalties, usable ground)
//   pick()   <- analyze.pick_candidates (6 m smoothing, peak_local_max with labels, snap, spread, zones)
"use strict";

const CougarKernel = (() => {
  const FACTORS = ["wind", "edges", "pinch", "water"]; // stacked; travel is added in but not stacked

  function ramp(a, lo, hi) {
    const t = (a - lo) / (hi - lo);
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  // scipy.ndimage.gaussian_filter1d weights (truncate 4.0)
  function gaussWeights(sigma) {
    const radius = Math.floor(4 * sigma + 0.5);
    const w = new Float64Array(2 * radius + 1);
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      w[i + radius] = Math.exp((-0.5 * i * i) / (sigma * sigma));
      sum += w[i + radius];
    }
    for (let i = 0; i < w.length; i++) w[i] /= sum;
    return { w, radius };
  }

  // scipy's mode="reflect" (d c b a | a b c d | d c b a)
  function reflect(i, n) {
    if (n === 1) return 0;
    const p = 2 * n;
    i = ((i % p) + p) % p;
    return i < n ? i : p - 1 - i;
  }

  // scipy.ndimage.gaussian_filter(a, sigma, mode="reflect"): rows (axis 0) first, then columns
  function gaussian(src, H, W, sigma) {
    const { w, radius } = gaussWeights(sigma);
    const tmp = new Float32Array(H * W);
    const out = new Float32Array(H * W);
    const idx = new Int32Array(Math.max(H, W) + 2 * radius);
    for (let i = 0; i < H + 2 * radius; i++) idx[i] = reflect(i - radius, H);
    const acc = new Float64Array(W);
    for (let r = 0; r < H; r++) {
      acc.fill(0);
      for (let k = 0; k < w.length; k++) {
        const wk = w[k], base = idx[r + k] * W;
        for (let c = 0; c < W; c++) acc[c] += wk * src[base + c];
      }
      tmp.set(acc, r * W);
    }
    for (let i = 0; i < W + 2 * radius; i++) idx[i] = reflect(i - radius, W);
    for (let r = 0; r < H; r++) {
      const row = r * W;
      for (let c = 0; c < W; c++) {
        let s = 0;
        for (let k = 0; k < w.length; k++) s += w[k] * tmp[row + idx[c + k]];
        out[row + c] = s;
      }
    }
    return out;
  }

  // analyze.combine's score (0-100) from the weighted layers L (wind, edges, pinch, water, travel, habitat =
  // habitat x season), weights w, habitat options h and the cell size res (m)
  function score(L, H, W, w, h, res) {
    const n = H * W;
    const s = new Float32Array(n);
    const m = w.stack_multiplier - 1;
    for (let i = 0; i < n; i++) {
      let spot =
        w.wind * L.wind[i] + w.edges * L.edges[i] + w.pinch * L.pinch[i] + w.water * L.water[i] + w.travel * L.travel[i];
      for (const f of FACTORS) spot *= 1 + m * ramp(L[f][i], w.stack_from, w.stack_to);
      s[i] = spot * L.habitat[i];
    }
    const z = h.zone_m > 0 ? gaussian(s, H, W, h.zone_m / res) : s;
    const top =
      (w.wind + w.edges + w.pinch + w.water + w.travel) *
      Math.pow(w.stack_multiplier, 4) *
      (1 + h.edge_floor) *
      (1 + h.water_floor);
    const k = top > 0 ? (100 * h.score_scale) / top : 0;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = k * z[i];
      out[i] = v < 0 ? 0 : v > 100 ? 100 : v;
    }
    return out;
  }

  // the ranked score: score x the site penalties that are on, 0 off the usable ground
  function final(sc, L, on) {
    const n = sc.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (!L.usable[i]) continue;
      let v = sc[i];
      if (on.paved) v *= L.paved[i];
      if (on.houses) v *= L.houses[i];
      if (on.recreation) v *= L.recreation[i];
      out[i] = v;
    }
    return out;
  }

  // ndimage.maximum_filter with a (2d+1) square and mode="nearest" (separable)
  function maxFilter(a, H, W, d) {
    const tmp = new Float32Array(H * W);
    const out = new Float32Array(H * W);
    for (let r = 0; r < H; r++) {
      const row = r * W;
      for (let c = 0; c < W; c++) {
        let m = -Infinity;
        const c0 = Math.max(0, c - d), c1 = Math.min(W - 1, c + d);
        for (let k = c0; k <= c1; k++) if (a[row + k] > m) m = a[row + k];
        tmp[row + c] = m;
      }
    }
    const m = new Float32Array(W);
    for (let r = 0; r < H; r++) {
      m.fill(-Infinity);
      const r0 = Math.max(0, r - d), r1 = Math.min(H - 1, r + d);
      for (let k = r0; k <= r1; k++) {
        const base = k * W;
        for (let c = 0; c < W; c++) if (tmp[base + c] > m[c]) m[c] = tmp[base + c];
      }
      out.set(m, r * W);
    }
    return out;
  }

  // analyze.spread: indices to keep (best first), at most perZone within radius of each other, n in all
  function spread(xy, perZone, radius, n) {
    const picked = [];
    for (let i = 0; i < xy.length; i++) {
      let near = 0;
      for (const j of picked) if (Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) < radius) near++;
      if (near >= perZone) continue;
      picked.push(i);
      if (picked.length >= n) break;
    }
    return picked;
  }

  // analyze.zones: zone number (1, 2, ...) of each point
  function zones(xy, radius) {
    const zone = xy.map((_, i) => i);
    for (let i = 0; i < xy.length; i++) {
      for (let j = 0; j < i; j++) {
        if (Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) < radius) {
          zone[i] = zone[j];
          break;
        }
      }
    }
    const map = new Map();
    return zone.map((z) => {
      if (!map.has(z)) map.set(z, map.size + 1);
      return map.get(z);
    });
  }

  // analyze.pick_candidates on the page grid: [{row, col, score, rank, zone}]
  function pick(fin, usable, H, W, res, p) {
    const n = H * W;
    let any = false;
    for (let i = 0; i < n; i++) if (usable[i]) { any = true; break; }
    if (!any) return [];
    const sm = gaussian(fin, H, W, Math.max(0.5, 6 / res));
    const img = new Float32Array(n);
    let best = -Infinity;
    for (let i = 0; i < n; i++) {
      img[i] = usable[i] ? sm[i] : -Infinity;
      if (usable[i] && sm[i] > best) best = sm[i];
    }
    const thr = Math.min(p.peak_min, Math.max(p.peak_floor, p.peak_rel * best));
    const d = Math.max(1, Math.round(p.spacing_m / res));
    const mx = maxFilter(img, H, W, d);
    let peaks = [];
    for (let i = 0; i < n; i++) if (usable[i] && img[i] === mx[i] && img[i] > thr) peaks.push(i);
    peaks.sort((a, b) => img[b] - img[a] || a - b); // highest first, stable (row-major ties)
    // skimage's ensure_spacing: greedy, Chebyshev distance < d, at most n x 8
    const maxOut = p.n * 8;
    const kept = [];
    for (const i of peaks) {
      const r = Math.floor(i / W), c = i % W;
      let ok = true;
      for (const j of kept) {
        if (Math.max(Math.abs(Math.floor(j / W) - r), Math.abs((j % W) - c)) < d) { ok = false; break; }
      }
      if (ok) kept.push(i);
      if (kept.length >= maxOut) break;
    }
    // snap to the best cell within ~10 m
    const k = Math.max(1, Math.floor(10 / res));
    let cands = kept.map((i) => {
      const pr = Math.floor(i / W), pc = i % W;
      let br = pr, bc = pc, bv = -Infinity;
      for (let r = Math.max(pr - k, 0); r <= Math.min(pr + k, H - 1); r++) {
        for (let c = Math.max(pc - k, 0); c <= Math.min(pc + k, W - 1); c++) {
          if (fin[r * W + c] > bv) { bv = fin[r * W + c]; br = r; bc = c; }
        }
      }
      return { row: br, col: bc, score: bv };
    });
    cands = cands.map((c, i) => [c, i]).sort((a, b) => b[0].score - a[0].score || a[1] - b[1]).map((x) => x[0]);
    const xy = cands.map((c) => [(c.col + 0.5) * res, -(c.row + 0.5) * res]);
    const keep = spread(xy, p.per_zone, p.zone_radius_m, p.n);
    const z = zones(keep.map((i) => xy[i]), p.zone_radius_m);
    return keep.map((i, j) => ({ ...cands[i], rank: j + 1, zone: z[j] }));
  }

  // everything for one set of slider values: weights w, penalties on {paved, houses, recreation}, spot count n
  function run(L, data, w, on, n) {
    const { height: H, width: W, res } = data.grid;
    const sc = score(L, H, W, w, data.habitat, res);
    const fin = final(sc, L, on);
    const spots = pick(fin, L.usable, H, W, res, { ...data.pick, n });
    return { score: sc, final: fin, spots };
  }

  // page cell centre -> [lat, lon] (explore.georef's forward fit)
  function cellLatLon(geo, H, W, row, col) {
    const u = col / W - 0.5, v = row / H - 0.5;
    const t = [1, u, v, u * u, u * v, v * v];
    let lon = 0, lat = 0;
    for (let i = 0; i < 6; i++) { lon += geo.fwd[0][i] * t[i]; lat += geo.fwd[1][i] * t[i]; }
    return [lat, lon];
  }

  // lat/lon -> fractional page [row, col] (the inverse fit)
  function latLonCell(geo, lat, lon) {
    const [west, south, east, north] = geo.bounds;
    const u = (lon - west) / (east - west) - 0.5, v = (north - lat) / (north - south) - 0.5;
    const t = [1, u, v, u * u, u * v, v * v];
    let c = 0, r = 0;
    for (let i = 0; i < 6; i++) { c += geo.inv[0][i] * t[i]; r += geo.inv[1][i] * t[i]; }
    return [r, c];
  }

  return { ramp, gaussian, score, final, maxFilter, pick, run, cellLatLon, latLonCell };
})();

if (typeof module !== "undefined") module.exports = CougarKernel;
