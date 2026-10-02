// MapLibre's worker, pre-bundled into one script (vite.worker.config.ts) and inlined, so the page needs no files
// beside it: MapLibre loads it from a blob URL.
import { setWorkerUrl } from "maplibre-gl";
import source from "../../build/maplibre-worker.js?raw";

setWorkerUrl(URL.createObjectURL(new Blob([source], { type: "text/javascript" })));
