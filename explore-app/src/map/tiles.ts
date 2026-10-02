// Map tiles: Esri imagery, topo and reference overlays (the KMZ's imagery too), and AWS Terrain Tiles for relief and
// 3D. All free, no key.
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";

export const TILES = {
  imagery: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
  topo: `${ESRI}/World_Topo_Map/MapServer/tile/{z}/{y}/{x}`,
  roads: `${ESRI}/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}`,
  labels: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
  dem: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
};

export const ATTRIBUTION =
  'Imagery and maps &copy; <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a>, Maxar, Earthstar ' +
  "Geographics, USGS and the GIS User Community. Elevation: AWS Terrain Tiles (USGS 3DEP, SRTM).";

export type Base = "satellite" | "hybrid" | "topo";

// one tile at zoom z around a point, as a thumbnail for the layer picker
export function tileAt(template: string, lat: number, lon: number, z: number): string {
  const n = 2 ** z;
  const x = Math.floor(((lon + 180) / 360) * n);
  const r = (lat * Math.PI) / 180;
  const y = Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n);
  return template.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y));
}
