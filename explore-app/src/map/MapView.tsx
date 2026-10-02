import "./workerUrl";
import * as maplibregl from "maplibre-gl";
import type { ImageSource, LngLatBoundsLike, MapLayerMouseEvent, MapMouseEvent, StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { FactorMap, Payload, Spot } from "../types";
import { ATTRIBUTION, TILES, type Base } from "./tiles";

export interface MapLayers {
  base: Base;
  heat: boolean;
  factor: FactorMap | null;
  public: boolean;
  private: boolean;
  reach: boolean;
  worn: boolean;
  routes: boolean;
  airflow: boolean;
  saddles: boolean;
  relief: boolean;
  terrain: boolean;
  ghosts: boolean;
  pins: boolean;
  outline: boolean;
}

export type Kind = "pub" | "priv";
export interface Selection {
  kind: Kind;
  rank: number;
}

interface Props {
  data: Payload;
  spots: Spot[];
  kind: Kind; // which land the spots are on
  heat?: ImageBitmap;
  factorImg?: ImageBitmap;
  landImg?: ImageBitmap;
  priv: boolean;
  layers: MapLayers;
  selected: Selection | null;
  hover: Selection | null;
  onSelect: (s: Selection | null) => void;
  card: ReactNode;
  padding: { left: number; bottom: number };
  onReady: (api: MapApi) => void;
  onError: (msg: string) => void;
}

export interface MapApi {
  fit: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  flyTo: (lat: number, lon: number) => void;
}

const lonlat = (ring: [number, number][]) => ring.map(([lat, lon]) => [lon, lat]);
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function areaBounds(data: Payload): [[number, number], [number, number]] {
  const [west, south, east, north] = data.geo.bounds;
  if (!data.outline.length) return [[west, south], [east, north]];
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const ring of data.outline)
    for (const [lat, lon] of ring) {
      w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat);
    }
  return [[w, s], [e, n]];
}

function style(data: Payload): StyleSpecification {
  const [west, south, east, north] = data.geo.bounds;
  const rings = data.outline.length
    ? data.outline.map(lonlat)
    : [[[west, south], [east, south], [east, north], [west, north], [west, south]]];
  const feats = data.features ?? { worn: [], routes: [], airflow: [], saddles: [] };
  const blank = () => ({
    type: "image" as const,
    url: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
    coordinates: [[west, north], [east, north], [east, south], [west, south]] as [[number, number], [number, number], [number, number], [number, number]],
  });
  const lines = (items: [[number, number][], Record<string, unknown>][]) => ({
    type: "geojson" as const,
    data: {
      type: "FeatureCollection" as const,
      features: items.map(([coords, properties]) => ({
        type: "Feature" as const,
        properties,
        geometry: { type: "LineString" as const, coordinates: coords },
      })),
    },
  });
  const raster = (url: string, maxzoom = 19) => ({ type: "raster" as const, tiles: [url], tileSize: 256, maxzoom });
  return {
    version: 8,
    sources: {
      imagery: { ...raster(TILES.imagery), attribution: ATTRIBUTION },
      topo: raster(TILES.topo),
      roads: raster(TILES.roads),
      labels: raster(TILES.labels),
      dem: { type: "raster-dem", tiles: [TILES.dem], tileSize: 256, maxzoom: 15, encoding: "terrarium" },
      demShade: { type: "raster-dem", tiles: [TILES.dem], tileSize: 256, maxzoom: 15, encoding: "terrarium" },
      mask: {
        type: "geojson",
        data: {
          type: "Feature",
          properties: {},
          geometry: { type: "Polygon", coordinates: [[[west - 3, south - 3], [east + 3, south - 3], [east + 3, north + 3], [west - 3, north + 3], [west - 3, south - 3]], ...rings] },
        },
      },
      outline: {
        type: "geojson",
        data: { type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: rings } },
      },
      heat: blank(),
      factor: blank(),
      land: blank(),
      worn: lines(feats.worn.map((w) => [w.coords, { mapped: w.mapped }])),
      routes: lines(feats.routes.map((r) => [r.coords, { name: r.name }])),
      airflow: lines(feats.airflow.map((a) => [a, {}])),
      saddles: {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: feats.saddles.map((sd) => ({
            type: "Feature",
            properties: { rise: sd.rise_m },
            geometry: { type: "Point", coordinates: sd.coords },
          })),
        },
      },
      pins: {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: data.pins.map((p) => ({
            type: "Feature",
            properties: { name: p.name, kind: p.kind },
            geometry: { type: "Point", coordinates: [p.lon, p.lat] },
          })),
        },
      },
    },
    layers: [
      { id: "imagery", type: "raster", source: "imagery", paint: { "raster-fade-duration": 150 } },
      { id: "topo", type: "raster", source: "topo", layout: { visibility: "none" } },
      {
        id: "relief",
        type: "hillshade",
        source: "demShade",
        layout: { visibility: "none" },
        paint: { "hillshade-exaggeration": 0.45, "hillshade-shadow-color": "#1d2a22", "hillshade-highlight-color": "#fffbea" },
      },
      { id: "roads", type: "raster", source: "roads", layout: { visibility: "none" } },
      { id: "labels", type: "raster", source: "labels", layout: { visibility: "none" } },
      { id: "land", type: "raster", source: "land", layout: { visibility: "none" }, paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } },
      { id: "factor", type: "raster", source: "factor", layout: { visibility: "none" }, paint: { "raster-opacity": 0.95, "raster-resampling": "linear", "raster-fade-duration": 0 } },
      {
        id: "heat",
        type: "raster",
        source: "heat",
        layout: { visibility: "none" },
        paint: { "raster-opacity": 0.9, "raster-resampling": "linear", "raster-fade-duration": 0 },
      },
      { id: "mask", type: "fill", source: "mask", paint: { "fill-color": "#0d1310", "fill-opacity": 0.58 } },
      {
        id: "outline-halo",
        type: "line",
        source: "outline",
        paint: { "line-color": "#0d1310", "line-width": 4, "line-opacity": 0.35, "line-blur": 2 },
      },
      {
        id: "outline",
        type: "line",
        source: "outline",
        paint: { "line-color": "#f6efdc", "line-width": 1.6, "line-dasharray": [3, 2.5], "line-opacity": 0.95 },
      },
      {
        id: "airflow",
        type: "line",
        source: "airflow",
        layout: { visibility: "none", "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#bfe6ff", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1, 16, 2.2], "line-opacity": 0.9 },
      },
      {
        id: "worn-mapped",
        type: "line",
        source: "worn",
        filter: ["==", ["get", "mapped"], true],
        layout: { visibility: "none", "line-cap": "round" },
        paint: { "line-color": "#e4e1d8", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.8, 17, 2], "line-opacity": 0.7 },
      },
      {
        id: "worn",
        type: "line",
        source: "worn",
        filter: ["==", ["get", "mapped"], false],
        layout: { visibility: "none", "line-cap": "round" },
        paint: { "line-color": "#ff8a2b", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1.2, 17, 3], "line-opacity": 0.95 },
      },
      {
        id: "routes",
        type: "line",
        source: "routes",
        layout: { visibility: "none", "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#ffe45c", "line-width": 2.4, "line-dasharray": [1.5, 1.5], "line-opacity": 0.95 },
      },
      {
        id: "saddles",
        type: "circle",
        source: "saddles",
        layout: { visibility: "none" },
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 3.5, 17, 6],
          "circle-color": "#c9a2ec",
          "circle-stroke-color": "#1b1426",
          "circle-stroke-width": 1.5,
        },
      },
      {
        id: "pins",
        type: "circle",
        source: "pins",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 3.5, 17, 6.5],
          "circle-color": ["match", ["get", "kind"], "sign", "#f2a13a", "#43b7cf"],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1.5,
        },
      },
    ],
  };
}

// a marker element: the spot's rank (public) or P-number (private); ghosts are the model's spots, dashed rings
function markerEl(kind: Kind | "ghost", label: string, title: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `mk mk-${kind}`;
  if (kind !== "ghost") {
    el.innerHTML = `<span class="mk-face"></span>`;
    el.firstElementChild!.textContent = label;
    el.setAttribute("role", "button");
  }
  el.title = title;
  return el;
}

interface Live {
  marker: maplibregl.Marker;
  el: HTMLElement;
  raf: number;
}

function glide(m: Live, to: [number, number]) {
  const from = m.marker.getLngLat();
  cancelAnimationFrame(m.raf);
  if (reduced() || (Math.abs(from.lng - to[0]) < 1e-9 && Math.abs(from.lat - to[1]) < 1e-9)) {
    m.marker.setLngLat(to);
    return;
  }
  const t0 = performance.now(), dur = 420;
  const step = (t: number) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    m.marker.setLngLat([from.lng + (to[0] - from.lng) * e, from.lat + (to[1] - from.lat) * e]);
    if (k < 1) m.raf = requestAnimationFrame(step);
  };
  m.raf = requestAnimationFrame(step);
}

function metres(a: [number, number], b: [number, number]) {
  const k = 111320, dy = (a[1] - b[1]) * k, dx = (a[0] - b[0]) * k * Math.cos((a[1] * Math.PI) / 180);
  return Math.hypot(dx, dy);
}

export default function MapView(p: Props) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const live = useRef<Record<Kind, Live[]>>({ pub: [], priv: [] });
  const ghosts = useRef<maplibregl.Marker[]>([]);
  const popup = useRef<maplibregl.Popup | null>(null);
  const [cardNode] = useState(() => document.createElement("div"));
  const props = useRef(p);
  props.current = p;

  // the map, once
  useEffect(() => {
    const bounds = areaBounds(p.data);
    const [[w, s], [e, n]] = bounds;
    const padX = (e - w) * 0.7, padY = (n - s) * 0.7;
    const map = new maplibregl.Map({
      container: box.current!,
      style: style(p.data),
      bounds: bounds as LngLatBoundsLike,
      fitBoundsOptions: { padding: { top: 24, right: 24, left: 24 + p.padding.left, bottom: 24 + p.padding.bottom } },
      maxBounds: [[w - padX, s - padY], [e + padX, n + padY]],
      maxZoom: 19.5,
      maxPitch: 75,
      attributionControl: false,
      dragRotate: true,
      pitchWithRotate: true,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    map.on("load", () => {
      setReady(true);
      box.current?.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
      const fit = () => {
        const { left, bottom } = props.current.padding;
        map.fitBounds(bounds, { padding: { top: 40, right: 40, left: 40 + left, bottom: 40 + bottom }, duration: reduced() ? 0 : 700 });
      };
      map.setMinZoom(Math.max(0, map.getZoom() - 1.5));
      props.current.onReady({
        fit,
        zoomIn: () => map.zoomIn(),
        zoomOut: () => map.zoomOut(),
        flyTo: (lat, lon) => map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 15.5), duration: reduced() ? 0 : 900, essential: true }),
      });
    });
    map.on("error", (ev) => {
      const msg = String((ev as unknown as { error?: Error }).error?.message ?? "");
      if (/tile|fetch|load|network|status/i.test(msg)) props.current.onError("Some map tiles did not load. Check the internet connection; the spots still update.");
    });
    // the user's pins: name on hover
    const tip = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: "tip", offset: 10 });
    map.on("mouseenter", "pins", (ev: MapLayerMouseEvent) => {
      const f = ev.features?.[0];
      if (!f) return;
      map.getCanvas().style.cursor = "default";
      const kind = String(f.properties.kind).replace("_", " ");
      tip.setLngLat(ev.lngLat).setText(`${f.properties.name} (your ${kind} pin)`).addTo(map);
    });
    map.on("mouseleave", "pins", () => tip.remove());
    map.on("mouseenter", "saddles", (ev: MapLayerMouseEvent) => {
      const f = ev.features?.[0];
      if (f) tip.setLngLat(ev.lngLat).setText(`Saddle (${Math.round(Number(f.properties.rise))} m)`).addTo(map);
    });
    map.on("mouseleave", "saddles", () => tip.remove());
    map.on("mouseenter", "routes", (ev: MapLayerMouseEvent) => {
      const f = ev.features?.[0];
      if (f) tip.setLngLat(ev.lngLat).setText(`Walking route to the model's ${f.properties.name}`).addTo(map);
    });
    map.on("mouseleave", "routes", () => tip.remove());
    map.on("click", (ev: MapMouseEvent) => {
      if (!(ev.originalEvent.target as HTMLElement).closest(".mk")) props.current.onSelect(null);
    });
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(box.current!);
    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // base and overlays
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const vis = (id: string, on: boolean) => map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    const { base, relief, heat, outline, pins, terrain } = p.layers;
    vis("imagery", base !== "topo");
    vis("topo", base === "topo");
    vis("roads", base === "hybrid");
    vis("labels", base === "hybrid");
    vis("relief", relief);
    vis("heat", heat);
    vis("outline", outline);
    vis("outline-halo", outline);
    vis("pins", pins);
    vis("factor", !!p.layers.factor);
    vis("land", p.layers.public || p.layers.private || p.layers.reach);
    vis("worn", p.layers.worn);
    vis("worn-mapped", p.layers.worn);
    vis("routes", p.layers.routes);
    vis("airflow", p.layers.airflow);
    vis("saddles", p.layers.saddles);
    map.setPaintProperty("mask", "fill-color", base === "topo" ? "#f2f1ea" : "#0d1310");
    map.setPaintProperty("mask", "fill-opacity", base === "topo" ? 0.72 : 0.58);
    map.setPaintProperty("outline", "line-color", base === "topo" ? "#3a2a18" : "#f6efdc");
    const has3d = !!map.getTerrain();
    if (terrain && !has3d) {
      map.setTerrain({ source: "dem", exaggeration: 1.4 });
      map.easeTo({ pitch: 58, duration: reduced() ? 0 : 900 });
    } else if (!terrain && has3d) {
      map.setTerrain(null);
      map.easeTo({ pitch: 0, bearing: 0, duration: reduced() ? 0 : 700 });
    }
  }, [p.layers, ready]);

  // the heat image
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !p.heat) return;
    (map.getSource("heat") as ImageSource).updateImage({ image: p.heat } as never);
  }, [p.heat, ready]);

  // the factor and land maps (drawn by the scoring thread on request)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !p.factorImg) return;
    (map.getSource("factor") as ImageSource).updateImage({ image: p.factorImg } as never);
  }, [p.factorImg, ready]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !p.landImg) return;
    (map.getSource("land") as ImageSource).updateImage({ image: p.landImg } as never);
  }, [p.landImg, ready]);

  // the model's spots, as dashed rings
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    for (const g of ghosts.current) g.remove();
    ghosts.current = [];
    if (!p.layers.ghosts) return;
    const add = (list: Payload["model_spots"]) => {
      for (const m of list) {
        const el = markerEl("ghost", "", `The model's spot ${m.name}, score ${Math.round(m.score)}`);
        ghosts.current.push(new maplibregl.Marker({ element: el }).setLngLat([m.lon, m.lat]).addTo(map));
      }
    };
    add(p.priv ? (p.data.model_private_spots ?? []) : p.data.model_spots);
  }, [p.layers.ghosts, p.priv, ready, p.data]);

  // the live spots: each new spot takes over the nearest old marker and glides there; extras pop in, leftovers fade
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const sync = (kind: Kind, spots: Spot[]) => {
      const free: (Live | null)[] = live.current[kind].slice();
      const next: Live[] = [];
      for (const s of spots) {
        const to: [number, number] = [s.lon, s.lat];
        let bi = -1, bd = Infinity;
        free.forEach((m, j) => {
          if (!m) return;
          const ll = m.marker.getLngLat();
          const d = metres(to, [ll.lng, ll.lat]);
          if (d < bd) { bd = d; bi = j; }
        });
        const label = kind === "pub" ? String(s.rank) : `P${s.rank}`;
        const title = `${kind === "pub" ? "Spot" : "Private-land spot"} ${label}, score ${Math.round(s.score)}`;
        let m: Live;
        if (bi >= 0 && bd < 4000) {
          m = free[bi]!;
          free[bi] = null;
          glide(m, to);
        } else {
          const el = markerEl(kind, label, title);
          el.classList.add("mk-fresh");
          const marker = new maplibregl.Marker({ element: el }).setLngLat(to).addTo(map);
          m = { marker, el, raf: 0 };
          el.addEventListener("click", (ev) => {
            ev.stopPropagation();
            const r = Number(el.dataset.rank);
            props.current.onSelect({ kind, rank: r });
          });
        }
        m.el.dataset.rank = String(s.rank);
        m.el.firstElementChild!.textContent = label;
        m.el.title = title;
        m.el.style.zIndex = String(200 - s.rank + (kind === "pub" ? 100 : 0));
        m.el.setAttribute("aria-label", title);
        next.push(m);
      }
      for (const m of free) {
        if (!m) continue;
        cancelAnimationFrame(m.raf);
        m.el.classList.add("mk-leaving");
        setTimeout(() => m.marker.remove(), 220);
      }
      live.current[kind] = next;
    };
    sync("pub", p.kind === "pub" ? p.spots : []);
    sync("priv", p.kind === "priv" ? p.spots : []);
  }, [p.spots, p.kind, ready]);

  // selected and hovered marker classes, and the spot card popup
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    for (const kind of ["pub", "priv"] as Kind[]) {
      for (const m of live.current[kind]) {
        const r = Number(m.el.dataset.rank);
        m.el.classList.toggle("is-selected", p.selected?.kind === kind && p.selected.rank === r);
        m.el.classList.toggle("is-hover", p.hover?.kind === kind && p.hover.rank === r);
      }
    }
    const s = p.selected?.kind === p.kind ? p.spots.find((x) => x.rank === p.selected!.rank) : undefined;
    if (!s) {
      popup.current?.remove();
      popup.current = null;
      return;
    }
    if (!popup.current) {
      popup.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, maxWidth: "320px", offset: 24, className: "card-pop", focusAfterOpen: false })
        .setDOMContent(cardNode)
        .setLngLat([s.lon, s.lat])
        .addTo(map);
    } else {
      popup.current.setLngLat([s.lon, s.lat]);
    }
  }, [p.selected, p.hover, p.spots, p.kind, ready, cardNode]);

  return (
    <>
      <div ref={box} className="map" aria-label="Map of the area with the camera spots" role="region" />
      {createPortal(p.card, cardNode)}
    </>
  );
}
