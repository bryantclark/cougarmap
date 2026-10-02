// The sidebar's controls: share sliders, the mix bar, penalty sliders, a segmented picker and a switch.
import { useId, type CSSProperties, type ReactNode } from "react";

type Vars = CSSProperties & Record<`--${string}`, string | number>;

interface SliderProps {
  label: string;
  hint?: string;
  color: string;
  value: number;
  min: number;
  max: number;
  step: number;
  model: number; // where the model-default tick sits
  text: string; // the live value
  off?: boolean;
  onChange: (v: number) => void;
  onReset?: () => void;
  ariaText?: string;
}

export function Slider(p: SliderProps) {
  const id = useId();
  const pos = (v: number) => (v - p.min) / (p.max - p.min);
  const changed = Math.abs(p.value - p.model) > (p.max - p.min) / 400;
  const style: Vars = { "--c": p.color, "--p": pos(p.value), "--t": pos(p.model) };
  return (
    <div className={`slider${p.off ? " is-off" : ""}`} style={style}>
      <div className="slider-head">
        <label htmlFor={id}>
          <span className="swatch" aria-hidden />
          {p.label}
        </label>
        {changed && p.onReset ? (
          <button type="button" className="slider-reset" onClick={p.onReset} title="Back to the model's value">
            reset
          </button>
        ) : null}
        <output htmlFor={id} className="slider-val">
          {p.text}
        </output>
      </div>
      {p.hint ? <div className="slider-hint">{p.hint}</div> : null}
      <div className="track">
        <div className="track-fill" />
        <div className="track-tick" title="The model's value" />
        <input
          id={id}
          type="range"
          min={p.min}
          max={p.max}
          step={p.step}
          value={p.value}
          aria-valuetext={p.ariaText ?? p.text}
          onChange={(e) => p.onChange(Number(e.target.value))}
        />
      </div>
    </div>
  );
}

export function MixBar({ parts }: { parts: { key: string; label: string; color: string; share: number }[] }) {
  return (
    <div className="mix" role="img" aria-label={parts.map((p) => `${p.label} ${Math.round(p.share * 100)}%`).join(", ")}>
      {parts.map((p) => (
        <span key={p.key} className="mix-seg" style={{ flexGrow: p.share, background: p.color } as CSSProperties} title={`${p.label} ${Math.round(p.share * 100)}%`} />
      ))}
    </div>
  );
}

export function Segmented<T extends string | number>(p: {
  label: string;
  options: T[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="seg" role="radiogroup" aria-label={p.label}>
      {p.options.map((o) => (
        <button
          key={String(o)}
          type="button"
          role="radio"
          aria-checked={o === p.value}
          className={o === p.value ? "is-on" : ""}
          onClick={() => p.onChange(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export function Switch(p: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; hint?: string }) {
  const id = useId();
  return (
    <label className="switch-row" htmlFor={id}>
      <span className="switch-text">
        {p.children}
        {p.hint ? <small>{p.hint}</small> : null}
      </span>
      <span className="switch">
        <input id={id} type="checkbox" role="switch" checked={p.checked} onChange={(e) => p.onChange(e.target.checked)} />
        <i aria-hidden />
      </span>
    </label>
  );
}

export function Icon({ name }: { name: "chevron-left" | "sliders" | "layers" | "plus" | "minus" | "fit" | "close" | "copy" | "sun" | "moon" | "check" }) {
  const d: Record<string, ReactNode> = {
    "chevron-left": <path d="M14.5 6 8.5 12l6 6" />,
    sliders: (
      <>
        <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
        <circle cx="16" cy="7" r="2" />
        <circle cx="10" cy="17" r="2" />
      </>
    ),
    layers: (
      <>
        <path d="m12 4 8.5 4.5L12 13 3.5 8.5Z" />
        <path d="m3.5 12.5 8.5 4.5 8.5-4.5" />
        <path d="m3.5 16.5 8.5 4.5 8.5-4.5" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    minus: <path d="M5 12h14" />,
    fit: <path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    copy: (
      <>
        <rect x="9" y="9" width="11" height="11" rx="2" />
        <path d="M5 15V6a2 2 0 0 1 2-2h8" />
      </>
    ),
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" />
      </>
    ),
    moon: <path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10Z" />,
    check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  };
  return (
    <svg className="icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {d[name]}
    </svg>
  );
}
