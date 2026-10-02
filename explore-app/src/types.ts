// The area payload explore.py writes into the page (explore.payload).
import type { Weights } from "./kernel";

export const FACTOR_KEYS = ["wind", "edges", "pinch", "water", "travel"] as const;
export type FactorKey = (typeof FACTOR_KEYS)[number];
export const PENALTY_KEYS = ["paved", "houses", "recreation"] as const;
export type PenaltyKey = (typeof PENALTY_KEYS)[number];

export interface Layer {
  dtype: "uint8" | "uint16";
  scale: number;
  shape: [number, number];
  data: string;
}

export interface ModelSpot {
  rank: number;
  name: string;
  lat: number;
  lon: number;
  score: number;
  factors: Record<string, number>;
  reasons: string[];
}

export interface Payload {
  version: number;
  area: string;
  month: number;
  grid: { height: number; width: number; res: number; fine_res: number; block: number };
  geo: { fwd: number[][]; inv: number[][]; bounds: [number, number, number, number]; max_error_m: number };
  weights: Weights;
  habitat: { zone_m: number; score_scale: number; edge_floor: number; water_floor: number };
  pick: { n: number; spacing_m: number; per_zone: number; zone_radius_m: number };
  layers: Record<string, Layer>;
  outline: [number, number][][]; // rings of [lat, lon]
  pins: { name: string; kind: string; lat: number; lon: number }[];
  model_spots: ModelSpot[];
  model_private_spots?: ModelSpot[];
  features?: Features;
}

export interface Features {
  worn: { mapped: boolean; coords: [number, number][] }[]; // lon/lat
  routes: { name: string; coords: [number, number][] }[];
  airflow: [number, number][][];
  saddles: { rise_m: number; coords: [number, number] }[];
}

// the factor maps in the layer menu: payload layers drawn alone, in their slider colour
export const FACTOR_MAPS = ["wind", "edges", "pinch", "water", "travel", "edge_meadow", "habitat", "winter"] as const;
export type FactorMap = (typeof FACTOR_MAPS)[number];

// what the compute worker sends back for a spot
export interface Spot {
  rank: number;
  zone: number;
  score: number;
  lat: number;
  lon: number;
  factors: Record<FactorKey, number>;
  habitat: number; // the habitat x season multiplier, as a share of its top
  penalties: Record<PenaltyKey, number>; // the model's multipliers at the spot (1 = no cut)
}

export interface Settings {
  shares: Record<FactorKey, number>; // sum to 1
  stack: number;
  pens: Record<PenaltyKey, number>; // 0 ignore, 1 the model, 2 twice as harsh
  n: number;
  priv: boolean; // show the private-land spots instead of the public ones
}

export interface Result {
  id: number;
  spots: Spot[]; // the shown land's: public, or private (P1, P2...)
  ms: number;
  heat?: ImageBitmap;
}
