// Types for kernel.js (plain JS so node runs it in the Python tests without a build step).
export type Grid = Float32Array | Uint8Array;
export type Layers = Record<string, Grid>;
export interface Weights {
  wind: number;
  edges: number;
  pinch: number;
  water: number;
  travel: number;
  stack_multiplier: number;
  stack_from: number;
  stack_to: number;
}
export type Strengths = Record<"paved" | "houses" | "recreation", number | boolean>;
export interface KSpot {
  row: number;
  col: number;
  score: number;
  rank: number;
  zone: number;
}
export interface RunResult {
  score: Float32Array;
  final?: Float32Array;
  spots: KSpot[];
  finalPrivate?: Float32Array;
  privateSpots?: KSpot[];
}
export function final(sc: Float32Array, usable: Grid, L: Layers, on: Strengths): Float32Array;
export const FACTORS: string[];
export const WEIGHTED: string[];
export function engine(L: Layers, data: unknown): { run(w: Weights, on: Strengths, n: number, land?: "public" | "private" | "both"): RunResult };
export function cellLatLon(geo: unknown, H: number, W: number, row: number, col: number): [number, number];
export function latLonCell(geo: unknown, lat: number, lon: number): [number, number];
