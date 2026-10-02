// MapLibre runs its tile work in a module worker it loads from a file next to its script. The weights page is one
// HTML file, so the worker (and the code it shares with the map) is first bundled into one script here; the page
// inlines it and hands MapLibre a blob URL (src/map/workerUrl.ts).
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "build",
    emptyOutDir: true,
    lib: { entry: "node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs", formats: ["es"], fileName: () => "maplibre-worker.js" },
    rollupOptions: { output: { codeSplitting: false } },
    minify: true,
  },
});
