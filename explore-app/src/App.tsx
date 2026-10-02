import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import ComputeWorker from "./compute.worker?worker&inline";
import { Icon, MixBar, Segmented, Slider } from "./controls";
import LayersPanel from "./LayersPanel";
import MapView, { type Kind, type MapApi, type MapLayers, type Selection } from "./map/MapView";
import { FACTOR_KEYS, PENALTY_KEYS, type FactorKey, type FactorMap, type ModelSpot, type Payload, type PenaltyKey, type Result, type Settings, type Spot } from "./types";
import { modelShares, percents, sameShares, setShare } from "./weights";

const FACTORS: Record<FactorKey, { label: string; hint: string }> = {
  wind: { label: "Wind", hint: "air flowing the same way all day" },
  edges: { label: "Edges", hint: "timber beside openings" },
  pinch: { label: "Pinch points", hint: "saddles, cliff bases, banks" },
  water: { label: "Water", hint: "springs, seeps, small ponds" },
  travel: { label: "Travel lines", hint: "drainage bottoms, ridge crossings" },
};
const PENALTIES: Record<PenaltyKey, { label: string; hint: string }> = {
  paved: { label: "Paved roads", hint: "traffic, people, theft" },
  houses: { label: "Houses", hint: "people, dogs, camera theft" },
  recreation: { label: "Trailheads and camps", hint: "people and dogs at the spot" },
};
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const COUNTS = [5, 10, 15, 25];
const color = (k: string) => `var(--c-${k})`;

function metres(a: { lat: number; lon: number }, b: { lat: number; lon: number }) {
  const k = 111320, dy = (a.lat - b.lat) * k, dx = (a.lon - b.lon) * k * Math.cos((a.lat * Math.PI) / 180);
  return Math.hypot(dx, dy);
}

function nearest(s: Spot, model: ModelSpot[]) {
  let best: { m: ModelSpot; d: number } | null = null;
  for (const m of model) {
    const d = metres(s, m);
    if (!best || d < best.d) best = { m, d };
  }
  return best;
}

// how a spot compares with the model's list: the same place (and its rank change) or a new place
function where(s: Spot, model: ModelSpot[]) {
  const g = nearest(s, model);
  if (!g) return { same: false, text: "", move: "new" };
  if (g.d <= 30) {
    const d = g.m.rank - s.rank;
    return { same: true, text: `the model's ${g.m.name}`, move: d > 0 ? `▲${d}` : d < 0 ? `▼${-d}` : "" };
  }
  const d = g.d >= 1000 ? `${(g.d / 1000).toFixed(1)} km` : `${Math.round(g.d / 10) * 10} m`;
  return { same: false, text: `new, ${d} from ${g.m.name}`, move: "new" };
}

// what drives a spot at these weights: its two biggest contributions
function drivers(s: Spot, shares: Settings["shares"]) {
  const parts = FACTOR_KEYS.map((k) => [k, shares[k] * s.factors[k]] as const).sort((a, b) => b[1] - a[1]);
  const top = parts.filter(([, v]) => v > 0).slice(0, 2).map(([k]) => FACTORS[k].label.toLowerCase());
  if (!top.length) return "";
  const text = top.join(" and ");
  return text[0].toUpperCase() + text.slice(1);
}

const penText = (v: number) => (v <= 0.001 ? "ignored" : Math.abs(v - 1) < 0.001 ? "as modeled" : `${v.toFixed(2)}x`);

function defaults(data: Payload): Settings {
  return {
    shares: modelShares(data),
    stack: data.weights.stack_multiplier,
    pens: { paved: 1, houses: 1, recreation: 1 },
    n: data.pick.n,
    priv: false,
  };
}

const LAYERS: MapLayers = {
  base: "satellite", heat: true, factor: null, public: false, private: false, reach: false, worn: false, routes: false,
  airflow: false, saddles: false, relief: false, terrain: false, ghosts: true, pins: true, outline: true,
};
const STORE = "cougarmap.layers";
function remembered(): Partial<MapLayers> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}
// the factor maps' colours on the map: the bright (dark-theme) set, which reads over imagery in either theme
const MAP_RGB: Record<FactorMap, [number, number, number]> = {
  wind: [119, 180, 226], edges: [152, 199, 102], pinch: [191, 152, 223], water: [72, 193, 207], travel: [214, 182, 124],
  edge_meadow: [176, 222, 120], habitat: [163, 179, 148], winter: [236, 240, 255],
};

const wide = () => matchMedia("(min-width: 761px)").matches;

export default function App({ data }: { data: Payload }) {
  const model = useMemo(() => defaults(data), [data]);
  const [s, setS] = useState<Settings>(model);
  const [layers, setLayersState] = useState<MapLayers>(() => ({ ...LAYERS, ...remembered() }));
  const [factorImg, setFactorImg] = useState<ImageBitmap>();
  const [landImg, setLandImg] = useState<ImageBitmap>();
  const [result, setResult] = useState<Result | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hover, setHover] = useState<Selection | null>(null);
  const [open, setOpen] = useState(wide); // sidebar shown (desktop) or sheet pulled up (phone; it starts as a peek)
  const [isWide, setWide] = useState(wide);
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);
  const [copied, setCopied] = useState(false);
  const mapApi = useRef<MapApi | null>(null);
  const hasPrivateData = !!data.layers.usable_private;

  useEffect(() => {
    const mq = matchMedia("(min-width: 761px)");
    const f = () => setWide(mq.matches);
    mq.addEventListener("change", f);
    return () => mq.removeEventListener("change", f);
  }, []);
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }, [theme]);

  // ---- the scoring worker: one run in flight, the latest settings queued behind it -------------------------------
  const worker = useRef<Worker | null>(null);
  const flight = useRef({ busy: false, queued: null as null | (Settings & { heat: boolean }), id: 0 });
  const send = useCallback((msg: Settings & { heat: boolean }) => {
    const f = flight.current;
    if (!worker.current) return;
    if (f.busy) {
      f.queued = msg;
      return;
    }
    f.busy = true;
    worker.current.postMessage({ type: "run", id: ++f.id, ...msg });
  }, []);
  useEffect(() => {
    const w = new ComputeWorker();
    worker.current = w;
    flight.current = { busy: false, queued: null, id: 0 };
    w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "ready") {
        setStatus("ready");
        return;
      }
      if (m.type === "factor") {
        setFactorImg(m.bitmap);
        return;
      }
      if (m.type === "land") {
        setLandImg(m.bitmap);
        return;
      }
      flight.current.busy = false;
      if (m.type === "error") {
        setStatus("error");
        setError(m.message);
        return;
      }
      setResult(m as Result); // the old heat bitmap is left to the GC: MapLibre may still upload it
      const q = flight.current.queued;
      if (q) {
        flight.current.queued = null;
        send(q);
      }
    };
    w.onerror = (e) => {
      setStatus("error");
      setError(e.message || "the scoring thread stopped");
    };
    w.postMessage({ type: "init", data });
    return () => {
      w.terminate();
      worker.current = null;
      setStatus("loading");
    };
  }, [data, send]);
  useEffect(() => {
    if (status === "ready") send({ ...s, heat: layers.heat });
  }, [s, layers.heat, status, send]);
  useEffect(() => {
    if (status === "ready" && layers.factor) worker.current?.postMessage({ type: "factor", key: layers.factor, rgb: MAP_RGB[layers.factor] });
  }, [layers.factor, status]);
  useEffect(() => {
    if (status === "ready" && (layers.public || layers.private || layers.reach))
      worker.current?.postMessage({ type: "land", public: layers.public, private: layers.private, reach: layers.reach });
  }, [layers.public, layers.private, layers.reach, status]);
  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(layers));
    } catch {
      /* private window: the layers just aren't remembered */
    }
  }, [layers]);

  // ---- derived -----------------------------------------------------------------------------------------------
  const pct = percents(s.shares);
  const kind: Kind = s.priv ? "priv" : "pub";
  const spots = result?.spots ?? [];
  const modelList = s.priv ? (data.model_private_spots ?? []) : data.model_spots;
  const placed = spots.map((x) => ({ s: x, w: where(x, modelList) }));
  const isDefault =
    sameShares(s.shares, model.shares) &&
    Math.abs(s.stack - model.stack) < 1e-6 &&
    PENALTY_KEYS.every((k) => Math.abs(s.pens[k] - 1) < 1e-6) &&
    s.n === model.n;

  const update = (patch: Partial<Settings>) => setS((old) => ({ ...old, ...patch }));
  const setLayers = (l: Partial<MapLayers>) => setLayersState((old) => ({ ...old, ...l }));
  const reset = () => setS((old) => ({ ...model, priv: old.priv }));

  const select = (sel: Selection | null, fly = false) => {
    setSelected(sel);
    if (sel && fly) {
      const x = spots.find((q) => q.rank === sel.rank);
      if (x) mapApi.current?.flyTo(x.lat, x.lon);
      if (!isWide) setOpen(false);
    }
  };

  // the selected spot's card
  const sel = selected && selected.kind === kind ? placed.find((p) => p.s.rank === selected.rank) : undefined;
  useEffect(() => {
    if (selected && !sel) setSelected(null);
  }, [selected, sel]);

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked: the coordinates stay selectable */
    }
  };

  const card = sel ? (
    <article className="card" aria-label={`Spot ${sel.s.rank}`}>
      <header className="card-head">
        <span className={`card-badge badge-${selected!.kind}`}>{selected!.kind === "priv" ? `P${sel.s.rank}` : sel.s.rank}</span>
        <div className="card-title">
          <strong>{selected!.kind === "priv" ? "Private-land spot" : "Spot"} {selected!.kind === "priv" ? `P${sel.s.rank}` : sel.s.rank}</strong>
          <span>
            Zone {sel.s.zone}
            {sel.w.text ? `, ${sel.w.same ? sel.w.text.replace("the model's", "matches the model's") : sel.w.text}` : ""}
          </span>
        </div>
        <div className="card-score" title="Score, 0-100 (60+ is strong)">
          {Math.round(sel.s.score)}
        </div>
        <button type="button" className="icon-btn small card-close" onClick={() => setSelected(null)} aria-label="Close">
          <Icon name="close" />
        </button>
      </header>
      <div className="bars">
        {FACTOR_KEYS.map((k) => (
          <div key={k} className="bar" style={{ ["--c" as string]: color(k), ["--v" as string]: Math.min(1, sel.s.factors[k]) }}>
            <span>{FACTORS[k].label}</span>
            <b />
            <em>{sel.s.factors[k].toFixed(2)}</em>
          </div>
        ))}
        <div className="bar" style={{ ["--c" as string]: "var(--c-habitat)", ["--v" as string]: Math.min(1, sel.s.habitat) }}>
          <span>Habitat</span>
          <b />
          <em>{sel.s.habitat.toFixed(2)}</em>
        </div>
      </div>
      <p className="card-note">
        {(() => {
          const cuts = PENALTY_KEYS.filter((k) => sel.s.penalties[k] < 0.995 && s.pens[k] > 0).map(
            (k) => `${PENALTIES[k].label.toLowerCase()} x${Math.pow(sel.s.penalties[k], s.pens[k]).toFixed(2)}`,
          );
          return cuts.length ? `Score cut for ${cuts.join(", ")}.` : "No people nearby to cut the score.";
        })()}
      </p>
      <button type="button" className="coords" onClick={() => copy(`${sel.s.lat.toFixed(5)}, ${sel.s.lon.toFixed(5)}`)}>
        <span>{sel.s.lat.toFixed(5)}, {sel.s.lon.toFixed(5)}</span>
        <Icon name={copied ? "check" : "copy"} />
        <span className="sr">{copied ? "copied" : "copy coordinates"}</span>
      </button>
    </article>
  ) : null;

  // ---- the phone sheet: drag the handle between peek and open -------------------------------------------------
  const drag = useRef<{ y: number; moved: boolean } | null>(null);
  const dragged = useRef(false);
  const [dragDy, setDragDy] = useState(0);
  const onHandleDown = (e: RPointerEvent) => {
    drag.current = { y: e.clientY, moved: false };
    dragged.current = false;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers can't be captured; the drag still works inside the handle */
    }
  };
  const onHandleMove = (e: RPointerEvent) => {
    if (!drag.current) return;
    const dy = e.clientY - drag.current.y;
    if (Math.abs(dy) > 6) drag.current.moved = true;
    if (drag.current.moved) setDragDy(dy);
  };
  const onHandleUp = (e: RPointerEvent) => {
    const d = drag.current;
    drag.current = null;
    setDragDy(0);
    if (!d?.moved) return;
    dragged.current = true;
    const dy = e.clientY - d.y;
    if (dy < -40) setOpen(true);
    else if (dy > 40) setOpen(false);
  };
  const onHandleClick = () => {
    if (dragged.current) {
      dragged.current = false;
      return;
    }
    setOpen((o) => !o);
  };

  const [west, south, east, north] = data.geo.bounds;
  const mid = { lat: (south + north) / 2, lon: (west + east) / 2 };
  const fewer = spots.length < s.n && status === "ready" && result;
  const zoneKm = (data.pick.zone_radius_m / 1000).toFixed(1).replace(/\.0$/, "");

  return (
    <div className={`app${open ? " is-open" : " is-closed"}${isWide ? " is-wide" : " is-phone"}`}>
      <aside
        className="side"
        aria-label="Weights"
        style={!isWide && dragDy ? { transform: `translateY(calc(var(--sheet-y) + ${dragDy}px))`, transition: "none" } : undefined}
      >
        {!isWide ? (
          <div
            className="sheet-handle"
            onPointerDown={onHandleDown}
            onPointerMove={onHandleMove}
            onPointerUp={onHandleUp}
            onPointerCancel={onHandleUp}
            onClick={onHandleClick}
            role="button"
            tabIndex={0}
            aria-label={open ? "Collapse the weights" : "Expand the weights"}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setOpen((o) => !o)}
          >
            <span />
          </div>
        ) : null}
        <header className="side-head" onClick={!isWide && !open ? () => setOpen(true) : undefined}>
          <div className="brand">
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden className="brand-mark">
              <path d="M12 3c-1.6 2.6-4.5 3.6-6.5 3.2.2 5.6 2.8 10.4 6.5 13.8 3.7-3.4 6.3-8.2 6.5-13.8C16.5 6.6 13.6 5.6 12 3Z" fill="currentColor" />
            </svg>
            CougarMap
          </div>
          <div className="head-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : t === "light" ? null : matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark"))}
              aria-label="Switch light and dark"
              title="Light or dark"
            >
              <Icon name={theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "sun" : "moon"} />
            </button>
            {isWide ? (
              <button type="button" className="icon-btn" onClick={() => setOpen(false)} aria-label="Hide the sidebar" title="Hide the sidebar">
                <Icon name="chevron-left" />
              </button>
            ) : null}
          </div>
          <h1>{data.area}</h1>
          <p className="sub">
            Spots for {MONTHS[data.month - 1]}, scored on {Math.round(data.grid.res)} m cells
          </p>
          {!isWide ? (
            <div className="peek-mix">
              <MixBar parts={FACTOR_KEYS.map((k) => ({ key: k, label: FACTORS[k].label, color: color(k), share: s.shares[k] }))} />
            </div>
          ) : null}
        </header>

        <div className="side-scroll">
          <section className="sec" aria-labelledby="mix-h">
            <div className="sec-head">
              <h2 id="mix-h">What makes a spot</h2>
              {!isDefault ? (
                <button type="button" className="text-btn" onClick={reset}>
                  Reset to the model
                </button>
              ) : null}
            </div>
            <MixBar parts={FACTOR_KEYS.map((k) => ({ key: k, label: FACTORS[k].label, color: color(k), share: s.shares[k] }))} />
            <p className="sec-hint">Each factor's share of the score. Raising one lowers the others; the notch is the model's.</p>
            {FACTOR_KEYS.map((k) => (
              <Slider
                key={k}
                label={FACTORS[k].label}
                hint={FACTORS[k].hint}
                color={color(k)}
                value={s.shares[k] * 100}
                min={0}
                max={100}
                step={0.5}
                model={model.shares[k] * 100}
                text={s.shares[k] < 0.0005 ? "off" : `${pct[k]}%`}
                off={s.shares[k] < 0.0005}
                onChange={(v) => update({ shares: setShare(s.shares, k, v / 100) })}
              />
            ))}
            <Slider
              label="Stacking bonus"
              hint="extra for a spot where several factors are strong"
              color="var(--accent)"
              value={s.stack}
              min={1}
              max={2}
              step={0.05}
              model={model.stack}
              text={s.stack <= 1.0001 ? "off" : `${s.stack.toFixed(2)}x`}
              off={s.stack <= 1.0001}
              onChange={(v) => update({ stack: v })}
              onReset={() => update({ stack: model.stack })}
            />
          </section>

          <section className="sec" aria-labelledby="pen-h">
            <div className="sec-head">
              <h2 id="pen-h">People nearby</h2>
            </div>
            <p className="sec-hint">How hard each one cuts the score: 0 ignores it, the notch is the model, 2 is twice as harsh.</p>
            {PENALTY_KEYS.map((k) => (
              <Slider
                key={k}
                label={PENALTIES[k].label}
                hint={PENALTIES[k].hint}
                color="var(--c-people)"
                value={s.pens[k]}
                min={0}
                max={2}
                step={0.05}
                model={1}
                text={penText(s.pens[k])}
                off={s.pens[k] <= 0.001}
                onChange={(v) => update({ pens: { ...s.pens, [k]: v } })}
                onReset={() => update({ pens: { ...s.pens, [k]: 1 } })}
              />
            ))}
          </section>

          <section className="sec" aria-labelledby="spots-h">
            <div className="sec-head">
              <h2 id="spots-h">Spots</h2>
              <Segmented label="How many spots" options={COUNTS} value={s.n} onChange={(n) => update({ n })} />
            </div>
            {hasPrivateData ? (
              <div className="land-toggle">
                <Segmented
                  label="Which land"
                  options={["Public land", "Private land"]}
                  value={s.priv ? "Private land" : "Public land"}
                  onChange={(v) => {
                    setSelected(null);
                    update({ priv: v === "Private land" });
                  }}
                />
                <p className="sec-hint">
                  {s.priv
                    ? "P spots on private ground: your own land, or with the owner's permission."
                    : "Open-access public land within the walk limit, the main list."}
                </p>
              </div>
            ) : null}
            {status === "error" ? <p className="sec-hint" role="alert">The scoring stopped: {error}</p> : null}
            {result && !spots.length ? (
              <p className="sec-hint" aria-live="polite">
                No {s.priv ? "private" : "public"}-land spot scores with these weights. Turn a factor up or reset to the model.
              </p>
            ) : null}
            {fewer && spots.length ? (
              <p className="sec-hint">
                Only {spots.length} qualify: the model keeps {data.pick.per_zone} per {zoneKm} km hotspot zone and drops weak peaks.
              </p>
            ) : null}
            <SpotList kind={kind} items={placed} selected={selected} onSelect={(x) => select(x, true)} onHover={setHover} shares={s.shares} />
            <p className="fine">
              The page scores a coarser grid than the analysis and keeps its land and walking rules. Reasons, walk times and
              land names are in the Google Earth file.
            </p>
          </section>
        </div>
      </aside>

      <main className="stage">
        <MapView
          data={data}
          spots={spots}
          kind={kind}
          heat={result?.heat}
          factorImg={factorImg}
          landImg={landImg}
          priv={s.priv}
          layers={layers}
          selected={selected}
          hover={hover}
          onSelect={(x) => select(x)}
          card={card}
          padding={{ left: 0, bottom: isWide ? 0 : 158 }}
          onReady={(api) => (mapApi.current = api)}
          onError={(m) => setNotice(m)}
        />
        {isWide && !open ? (
          <button type="button" className="reopen" onClick={() => setOpen(true)} aria-label="Show the weights">
            <Icon name="sliders" />
            Weights
          </button>
        ) : null}
        <div className="zoom">
          <button type="button" className="icon-btn" onClick={() => mapApi.current?.zoomIn()} aria-label="Zoom in">
            <Icon name="plus" />
          </button>
          <button type="button" className="icon-btn" onClick={() => mapApi.current?.zoomOut()} aria-label="Zoom out">
            <Icon name="minus" />
          </button>
          <button type="button" className="icon-btn" onClick={() => mapApi.current?.fit()} aria-label="Fit the area" title="Fit the area">
            <Icon name="fit" />
          </button>
        </div>
        <LayersPanel layers={layers} set={setLayers} lat={mid.lat} lon={mid.lon} data={data} />
        {layers.factor ? <Legend factor={layers.factor} onClose={() => setLayers({ factor: null })} /> : null}
        {status === "loading" ? (
          <div className="loading" role="status">
            <span className="spinner" aria-hidden />
            Reading the area
          </div>
        ) : null}
        {notice ? (
          <div className="notice" role="status">
            {notice}
            <button type="button" className="icon-btn small" onClick={() => setNotice("")} aria-label="Dismiss">
              <Icon name="close" />
            </button>
          </div>
        ) : null}
      </main>
    </div>
  );
}

function SpotList(p: {
  kind: Kind;
  items: { s: Spot; w: { same: boolean; text: string; move: string } }[];
  selected: Selection | null;
  onSelect: (s: Selection) => void;
  onHover: (s: Selection | null) => void;
  shares: Settings["shares"];
}) {
  return (
    <ol className="spots">
      {p.items.map(({ s, w }) => {
        const on = p.selected?.kind === p.kind && p.selected.rank === s.rank;
        const part = (k: FactorKey) => p.shares[k] * s.factors[k]; // what each factor adds to this spot
        const total = FACTOR_KEYS.reduce((a, k) => a + part(k), 0) || 1;
        return (
          <li key={s.rank}>
            <button
              type="button"
              className={`spot${on ? " is-selected" : ""}`}
              onClick={() => p.onSelect({ kind: p.kind, rank: s.rank })}
              onMouseEnter={() => p.onHover({ kind: p.kind, rank: s.rank })}
              onMouseLeave={() => p.onHover(null)}
              onFocus={() => p.onHover({ kind: p.kind, rank: s.rank })}
              onBlur={() => p.onHover(null)}
            >
              <span className={`badge badge-${p.kind}`}>{p.kind === "priv" ? `P${s.rank}` : s.rank}</span>
              <span className="spot-mid">
                <span className="spot-mix" aria-hidden>
                  {FACTOR_KEYS.map((k) => (
                    <i key={k} style={{ flexGrow: part(k) / total, background: color(k) }} />
                  ))}
                </span>
                <span className="spot-where" title={w.text}>
                  {drivers(s, p.shares)}
                  {w.move ? <em className={`move move-${w.move === "new" ? "new" : w.move[0] === "▲" ? "up" : "down"}`}>{w.move}</em> : null}
                </span>
              </span>
              <span className="spot-score">{Math.round(s.score)}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

export const MAP_LABELS: Record<FactorMap, string> = {
  wind: "Wind",
  edges: "Edges",
  pinch: "Pinch points",
  water: "Water",
  travel: "Travel lines",
  edge_meadow: "Hunting edge",
  habitat: "Habitat",
  winter: "Winter ground",
};

function Legend({ factor, onClose }: { factor: FactorMap; onClose: () => void }) {
  const [r, g, b] = MAP_RGB[factor];
  return (
    <div className="legend" role="status">
      <span className="legend-name">{MAP_LABELS[factor]}</span>
      <span className="legend-ramp" style={{ background: `linear-gradient(to right, rgb(${r} ${g} ${b} / 0), rgb(${r} ${g} ${b}), rgb(${Math.round(r + (255 - r) * 0.55)} ${Math.round(g + (255 - g) * 0.55)} ${Math.round(b + (255 - b) * 0.55)}))` }} />
      <span className="legend-ends">
        <span>low</span>
        <span>high</span>
      </span>
      <button type="button" className="icon-btn small" onClick={onClose} aria-label="Hide the factor map">
        <Icon name="close" />
      </button>
    </div>
  );
}
