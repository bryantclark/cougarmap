// The factor sliders are shares of the spot score that always add up to 100%. That is exactly the model: its score
// divides by the sum of the weights (analyze.combine), so only their ratios ever mattered.
import { FACTOR_KEYS, type FactorKey, type Payload } from "./types";

export type Shares = Record<FactorKey, number>;

export function modelShares(data: Payload): Shares {
  const sum = FACTOR_KEYS.reduce((s, k) => s + data.weights[k], 0) || 1;
  return Object.fromEntries(FACTOR_KEYS.map((k) => [k, data.weights[k] / sum])) as Shares;
}

// set one share; the others keep their proportions and fill what is left (an even split when they were all 0)
export function setShare(shares: Shares, key: FactorKey, value: number): Shares {
  const v = Math.min(1, Math.max(0, value));
  const others = FACTOR_KEYS.filter((k) => k !== key);
  const rest = others.reduce((s, k) => s + shares[k], 0);
  const out = { ...shares, [key]: v };
  for (const k of others) out[k] = rest > 1e-9 ? (shares[k] * (1 - v)) / rest : (1 - v) / others.length;
  return out;
}

// whole percentages that add up to 100 (largest remainder)
export function percents(shares: Shares): Record<FactorKey, number> {
  const raw = FACTOR_KEYS.map((k) => shares[k] * 100);
  const floor = raw.map(Math.floor);
  let left = 100 - floor.reduce((a, b) => a + b, 0);
  const order = raw.map((r, i) => [r - floor[i], i] as const).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) {
    if (left <= 0) break;
    floor[i]++;
    left--;
  }
  return Object.fromEntries(FACTOR_KEYS.map((k, i) => [k, floor[i]])) as Record<FactorKey, number>;
}

export function sameShares(a: Shares, b: Shares): boolean {
  return FACTOR_KEYS.every((k) => Math.abs(a[k] - b[k]) < 5e-4);
}
