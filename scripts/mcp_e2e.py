"""End-to-end MCP check (live network): find_hotspots, then poll job_status the way an agent would.

Usage: uv run python scripts/mcp_e2e.py "<place>" <radius_km>
"""

from __future__ import annotations

import json
import sys
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _payload(res: Any) -> dict[str, Any]:
    return dict(json.loads(res.content[0].text))


def _report(result: dict[str, Any]) -> None:
    print("kmz:", result["kmz"])
    print("wind:", result["wind"])
    for b in result["blocks"]:
        print(" block", b["block"], b["land"], b["scout_score"], b["best_spot_score"])
    for c in result["top_spots"][:5]:
        reasons = "; ".join(c["reasons"])[:160]
        print(" ", c["name"], c["score"], c["walk_miles"], c["land"][:40], "|", reasons)


async def main(place: str, radius: float) -> None:
    params = StdioServerParameters(command="uv", args=["run", "--directory", ".", "cougarmap-mcp"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        print("instructions chars:", len(getattr(init, "instructions", "") or ""))
        tools = await session.list_tools()
        print("tools:", sorted(t.name for t in tools.tools))
        st = _payload(
            await session.call_tool("find_hotspots", {"location": place, "radius_km": radius, "wait_seconds": 5})
        )
        print("first:", st["state"], st.get("job_id"))
        while st["state"] in ("queued", "running"):
            st = _payload(await session.call_tool("job_status", {"job_id": st["job_id"], "wait_seconds": 50}))
            print(" ", st["state"], (st.get("progress") or [""])[-1])
        if st["state"] == "done":
            _report(st["result"])
        else:
            print(st)


if __name__ == "__main__":
    anyio.run(main, sys.argv[1], float(sys.argv[2]))
