import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import type { Payload } from "./types";
import "./styles.css";

const SUPPORTED = 2; // explore.payload's version this app reads
const root = createRoot(document.getElementById("root")!);

function fail(msg: string) {
  root.render(
    <div className="fatal" role="alert">
      <h1>The weights page could not open</h1>
      <p>{msg}</p>
    </div>,
  );
}

async function load(): Promise<Payload> {
  const text = document.getElementById("cougarmap-data")?.textContent?.trim() ?? "";
  // the CLI swaps the marker in the tag for the area's data; `npm run dev` serves a payload file instead
  if (!text.startsWith("{")) {
    const r = await fetch("/__payload.json");
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }
  return JSON.parse(text);
}

(async () => {
  try {
    if (typeof DecompressionStream === "undefined" || typeof Worker === "undefined") {
      throw new Error("This browser is too old for this page. Open it in a current Chrome, Edge, Firefox or Safari.");
    }
    const data = await load();
    if (data.version !== SUPPORTED) {
      throw new Error(`This page reads area data version ${SUPPORTED}, the file has ${data.version}. Rerun with --interactive to rewrite it.`);
    }
    root.render(
      <StrictMode>
        <App data={data} />
      </StrictMode>,
    );
  } catch (e) {
    fail(String((e as Error).message ?? e));
  }
})();
