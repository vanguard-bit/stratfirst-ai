# Typical weekday resource footprint

Measured on the operator’s Linux host from **systemd user-unit accounting**
(`Consumed … memory peak` / wall / CPU) during live paper sessions around
**2026-08-28**, plus on-disk sizes as of **2026-08-29**. These are operational
observations for a Nifty-50 paper lab — not cloud billing estimates and not a
promise of constant load.

## Always-on

| Component | Typical | Cap / notes |
|---|---|---|
| `nse-trader-dashboard.service` | ~60–120 MB RSS; ~70 MB MemoryPeak observed; near-idle CPU | Serves local/Tailscale HTML only |
| Project `.venv` | ~0.8 GB on disk | Shared by all jobs |
| `data/store/` | ~0.8 GB (DuckDB ~0.4 GB + replay cache ~0.3 GB) | Grows with history; rotate manually |

## Weekday job samples (IST)

| Job | When | Wall clock (typ.) | Memory peak (typ.) | CPU time (typ.) |
|---|---|---|---|---|
| Fyers refresh | 08:30 | seconds | low tens of MB | seconds |
| Morning LLM extract (+ ≤2 MCP calls) | 08:40 | ~1 min (mostly network) | ~70 MB | ~4 s |
| Ingest + paper-live tick | every minute 09:10–15:35 | ~50–70 s | **~1.1–1.2 GB** | ~25–40 s |
| Paper MIS backup | 15:20 | &lt;1 s | ~100–140 MB | ~1–2 s |
| EOD (metrics, health, glance export) | 15:45 | ~30–40 s | **~0.7 GB** | ~35–40 s |

**Hard caps (systemd):** ingest `MemoryMax=2G`, paper `MemoryMax=1G` (see
`deploy/systemd/` and `config/ops.yaml` `resources`).

## Average-day interpretation

- **CPU:** Dominated by ~390 market-minute ingest ticks. Roughly **2–3 core-hours**
  of CPU per session day if each tick uses ~25 s CPU (host was not saturated;
  wall time overlaps I/O and websocket wait).
- **RAM:** Peak concurrent demand is essentially **one ingest tick (~1.2 GB)** plus
  the small always-on dashboard (~0.1 GB). Jobs are oneshots and exit when done.
- **Network:** Fyers websocket/quotes during session; Gemini once in the morning;
  MCP enrichment **≤2 calls/day and ≤40/month** (budgeted). Pages publish is a
  small git push of `docs/site/` after healthy EOD.
- **Disk growth:** Feature parquets and logs are small vs DuckDB/cache; retention
  reminder only — no auto-delete.

## Offline / judge path

`uv sync` + `python main.py test` + `python main.py demo` need no broker and no
always-on services. Demo writes only under `data/demo/` (gitignored runtime).

Foreground heavies (`backfill-history`, `meta-train`) are **not** on the weekday
timers and can use multiple cores for hours — treat them as batch, not “average day.”
