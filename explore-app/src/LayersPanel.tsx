// The map's layer picker, Google-Maps style: a thumbnail chip that opens a panel of base maps and every layer the
// Google Earth file has (factor maps, land, worn trails, routes, air flow, saddles).
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { MAP_LABELS } from "./App";
import { Icon, Switch } from "./controls";
import type { MapLayers } from "./map/MapView";
import { TILES, tileAt, type Base } from "./map/tiles";
import { FACTOR_MAPS, type FactorMap, type Payload } from "./types";

const BASES: { key: Base; label: string; tiles: string[] }[] = [
  { key: "satellite", label: "Satellite", tiles: [TILES.imagery] },
  { key: "hybrid", label: "Hybrid", tiles: [TILES.imagery, TILES.roads, TILES.labels] },
  { key: "topo", label: "Topo", tiles: [TILES.topo] },
];

const FACTOR_COLOR: Record<FactorMap, string> = {
  wind: "var(--c-wind)",
  edges: "var(--c-edges)",
  pinch: "var(--c-pinch)",
  water: "var(--c-water)",
  travel: "var(--c-travel)",
  edge_meadow: "var(--c-edges)",
  habitat: "var(--c-habitat)",
  winter: "var(--muted)",
};

function Thumb({ tiles, lat, lon }: { tiles: string[]; lat: number; lon: number }) {
  return (
    <span className="thumb" aria-hidden>
      {tiles.map((t) => (
        <img key={t} src={tileAt(t, lat, lon, 13)} alt="" loading="lazy" draggable={false} />
      ))}
    </span>
  );
}

function Swatch({ kind }: { kind: string }) {
  return <span className={`lg lg-${kind}`} aria-hidden />;
}

export default function LayersPanel(p: { layers: MapLayers; set: (l: Partial<MapLayers>) => void; lat: number; lon: number; data: Payload }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  const L = p.layers;
  const f = p.data.features;
  const maps = FACTOR_MAPS.filter((k) => p.data.layers[k]);
  const next = BASES.find((b) => b.key === (L.base === "topo" ? "satellite" : "topo"))!;
  const count = (n: number | undefined, what: string) => (n ? `${n} ${what}` : `none in this area`);
  return (
    <div className="layers" ref={ref}>
      {open ? (
        <div className="layers-panel" role="dialog" aria-label="Map layers">
          <div className="layers-title">
            <span>Map type</span>
            <button type="button" className="icon-btn small" onClick={() => setOpen(false)} aria-label="Close map layers">
              <Icon name="close" />
            </button>
          </div>
          <div className="bases" role="radiogroup" aria-label="Map type">
            {BASES.map((b) => (
              <button
                key={b.key}
                type="button"
                role="radio"
                aria-checked={L.base === b.key}
                className={`base${L.base === b.key ? " is-on" : ""}`}
                onClick={() => p.set({ base: b.key })}
              >
                <Thumb tiles={b.tiles} lat={p.lat} lon={p.lon} />
                <span>{b.label}</span>
              </button>
            ))}
          </div>

          <div className="layers-group">
            <Switch checked={L.heat} onChange={(heat) => p.set({ heat })} hint="the camera-spot score for these weights">
              <Swatch kind="heat" />
              Score heat
            </Switch>
          </div>

          <div className="layers-title">
            <span>Factor map</span>
          </div>
          <div className="chips" role="radiogroup" aria-label="Factor map">
            {maps.map((k) => (
              <button
                key={k}
                type="button"
                role="radio"
                aria-checked={L.factor === k}
                className={`chip${L.factor === k ? " is-on" : ""}`}
                style={{ "--c": FACTOR_COLOR[k] } as CSSProperties}
                onClick={() => p.set({ factor: L.factor === k ? null : k })}
              >
                <i aria-hidden />
                {MAP_LABELS[k]}
              </button>
            ))}
          </div>

          <div className="layers-title">
            <span>Land</span>
          </div>
          <div className="layers-group">
            <Switch checked={L.public} onChange={(v) => p.set({ public: v })}>
              <Swatch kind="public" />
              Public land
            </Switch>
            <Switch checked={L.private} onChange={(v) => p.set({ private: v })}>
              <Swatch kind="private" />
              Private land
            </Switch>
            <Switch checked={L.reach} onChange={(v) => p.set({ reach: v })} hint="within this run's walk limit of a road">
              <Swatch kind="reach" />
              Within walking range
            </Switch>
          </div>

          <div className="layers-title">
            <span>On the ground</span>
          </div>
          <div className="layers-group">
            <Switch checked={L.worn} onChange={(v) => p.set({ worn: v })} hint={f?.worn.length ? "from lidar: orange on no map (game trails), grey mapped" : count(0, "")}>
              <Swatch kind="worn" />
              Worn trails
            </Switch>
            <Switch checked={L.routes} onChange={(v) => p.set({ routes: v })} hint="from the road to each of the model's spots">
              <Swatch kind="routes" />
              Walking routes
            </Switch>
            <Switch checked={L.airflow} onChange={(v) => p.set({ airflow: v })} hint="which way the air drains at dawn and dusk">
              <Swatch kind="airflow" />
              Dawn and dusk air flow
            </Switch>
            <Switch checked={L.saddles} onChange={(v) => p.set({ saddles: v })} hint={count(f?.saddles.length, "low crossings between ridges")}>
              <Swatch kind="saddles" />
              Saddles
            </Switch>
            {p.data.pins.length ? (
              <Switch checked={L.pins} onChange={(pins) => p.set({ pins })} hint="water and sign from your Google Earth file">
                <Swatch kind="pins" />
                Your pins
              </Switch>
            ) : null}
          </div>

          <div className="layers-title">
            <span>Map</span>
          </div>
          <div className="layers-group">
            <Switch checked={L.ghosts} onChange={(ghosts) => p.set({ ghosts })} hint="dashed rings: the analysis's own picks">
              <Swatch kind="ghost" />
              The model's spots
            </Switch>
            <Switch checked={L.relief} onChange={(relief) => p.set({ relief })} hint="ridges and draws in shade">
              Relief shading
            </Switch>
            <Switch checked={L.terrain} onChange={(terrain) => p.set({ terrain })} hint="tilt with right-drag or two fingers">
              3D terrain
            </Switch>
            <Switch checked={L.outline} onChange={(outline) => p.set({ outline })}>
              Area outline
            </Switch>
          </div>
        </div>
      ) : null}
      <button type="button" className="layers-chip" aria-expanded={open} aria-label="Map layers" onClick={() => setOpen((o) => !o)}>
        <Thumb tiles={next.tiles} lat={p.lat} lon={p.lon} />
        <span className="layers-chip-label">
          <Icon name="layers" />
          Layers
        </span>
      </button>
    </div>
  );
}
