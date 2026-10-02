// The weights app, built into one HTML file (dist/index.html) that the CLI fills with an area's data
// (explore.py: render). `npm run dev` serves it with the payload from COUGARMAP_DEV_PAYLOAD (a JSON file outside
// the repo, e.g. pulled out of a written explore.html), so private areas never land in this folder.
import fs from "node:fs";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

function devPayload(): Plugin {
  return {
    name: "cougarmap-dev-payload",
    configureServer(server) {
      server.middlewares.use("/__payload.json", (_req, res) => {
        const path = process.env.COUGARMAP_DEV_PAYLOAD;
        if (!path || !fs.existsSync(path)) {
          res.statusCode = 404;
          res.end("set COUGARMAP_DEV_PAYLOAD to a payload JSON file");
          return;
        }
        res.setHeader("Content-Type", "application/json");
        fs.createReadStream(path).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), viteSingleFile(), devPayload()],
  worker: { format: "es" },
  server: { host: "127.0.0.1", port: 5188, strictPort: true },
  build: { target: "es2022", assetsInlineLimit: 100_000_000, chunkSizeWarningLimit: 4000, reportCompressedSize: false },
});
