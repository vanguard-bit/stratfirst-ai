# Public-release checklist for StratFirst AI

This is the ordered handoff for a Cursor agent preparing the project for a public Razorpay Buildathon submission.

## Instructions for the agent

Work from the primary workspace only. Complete tasks in order, keep the system paper-only, and preserve all claim boundaries in `README.md`.

Do **not**:

- place or add live broker orders;
- change `meta_allocator.mode` away from `rules`;
- promote LightGBM out of shadow;
- weaken a safety check to make a demo pass;
- commit `.env`, tokens, runtime logs, private paths, or large market datasets;
- create a Git commit, GitHub repository, or push until the user explicitly requests it.

After each phase, run its acceptance commands. If a check fails, diagnose and fix the cause before continuing.

## Current pre-release baseline

- [ ] Confirm this workspace is still the intended primary copy.
- [ ] Read `README.md`, `docs/FAILURE_RECOVERY.md`, and `docs/paper/limitations-and-methods.md`.
- [ ] Run:

```bash
.venv/bin/python main.py test
.venv/bin/python main.py diagnose --json
.venv/bin/python main.py meta-status
```

Known baseline at the time this checklist was written:

- 299 tests collected; one environment-isolation test fails when real Fyers credentials are present.
- Runtime diagnosis passes but warns about apparent allocator concentration/streak and absent standard trade parquet files.
- The folder is not yet a Git repository.
- Live artifacts are local and must not be copied wholesale into the public repository.

---

## Phase 1 — Make verification clean and environment-independent

### 1. Fix the credential-isolation test

- [ ] Reproduce `tests/test_forward.py::test_ingest_fallback_without_fyers_creds` while the real `.env` exists.
- [ ] Confirm the cause: the test removes process variables, but Fyers auth reloads the project `.env`.
- [ ] Isolate the test from `.env` using a temporary empty env path or dependency injection. Do not rename/delete the user’s real `.env`.
- [ ] Ensure the test cannot trigger token remint, TOTP login, websocket access, or external network calls.
- [ ] Add/assert a regression that the fallback result is `placeholder`.

Acceptance:

```bash
.venv/bin/python -m pytest tests/test_forward.py -q
.venv/bin/python main.py test
```

Expected: all tests pass with real local credentials present.

### 2. Resolve misleading runtime warnings

- [ ] Investigate why health reports disabled `E1` at 100% average weight and `B1` as top for hundreds of consecutive “rebalances.”
- [ ] Determine whether repeated minute snapshots are being counted as daily rebalances or stale disabled strategies remain in history.
- [ ] Fix the metric semantics or allocator data—not merely the warning text.
- [ ] Investigate “No trade parquet files yet” versus the populated measure ledger. Either generate the standard judge-demo artifact or make health distinguish live capital trades from measurement fills.
- [ ] Preserve warnings when they indicate genuine concentration or missing execution evidence.

Acceptance:

```bash
.venv/bin/python main.py health
.venv/bin/python main.py diagnose --json
```

Expected: no unexplained warnings; any remaining warning is documented in `docs/results/RESULTS.md`.

---

## Phase 2 — Sanitize the repository boundary

### 3. Harden `.gitignore`

- [ ] Keep Python source under `data/`; do not ignore the entire directory.
- [ ] Ignore generated/private paths:

```gitignore
.env
.env.*
!.env.example
.venv/
venv/
.pytest_cache/
__pycache__/
*.py[cod]
*.duckdb
*.parquet
*.log
fyersDataSocket.log
data/store/
data/logs/
data/state/
data/archive/
.cursor/
docs/superpowers/
```

- [ ] Decide whether `data/fees/registry.json` is reproducible public seed data; include it only if its sources and license are clear.
- [ ] Add empty-directory placeholders only when runtime setup requires them.

### 4. Scan for credentials and private data

- [ ] Search all candidate public files for Gemini, Fyers, Tapetide, OAuth codes, JWTs, PINs, TOTP secrets, home paths, IPs, and account identifiers.
- [ ] Confirm `.env.example` contains placeholders only.
- [ ] Ensure MCP budget logs, sentiment parquet, Fyers logs, virtual books, dashboard payloads, and auth metadata are excluded.
- [ ] Search for `/home/loki`, Tailscale addresses, and machine-specific service paths.
- [ ] Do not print secret values into the agent transcript; report only affected filenames and key names.

Acceptance:

```bash
rg -n --hidden \
  -g '!.env' -g '!.venv/**' -g '!data/store/**' -g '!data/logs/**' \
  '(AIza[0-9A-Za-z_-]{20,}|tpt_[A-Za-z0-9_]{20,}|eyJ[A-Za-z0-9_-]{20,}|FYERS_PIN=[0-9]+|/home/loki)'
```

Expected: no real secret or private absolute path in public files. Placeholder/documentation matches must be reviewed manually.

---

## Phase 3 — Make the project portable

### 5. Remove hardcoded installation paths

- [ ] Replace `/home/loki/projects/nse-trader` in `deploy/systemd/*.service`.
- [ ] Prefer generated user units, an install script that substitutes the resolved project root, or systemd-safe environment/specifier usage.
- [ ] Keep `deploy/enable-user-timers.sh` idempotent.
- [ ] Ensure service `WorkingDirectory`, `.env`, Python executable, and log paths resolve from the actual clone.
- [ ] Remove private paths from user-facing docs and exported artifacts.

Acceptance:

```bash
rg -n '/home/loki|100\.108\.' README.md deploy docs/FAILURE_RECOVERY.md
```

Expected: no machine-specific paths.

### 6. Validate dependency setup from a clean environment

- [ ] Create a fresh temporary virtualenv outside the project.
- [ ] Install `requirements.txt`.
- [ ] Run the offline test/demo path with no `.env`.
- [ ] Confirm optional Fyers, Gemini, and Tapetide integrations fail with actionable messages or use documented fallbacks.
- [ ] Pin or constrain dependencies further only where a clean install exposes instability.

Acceptance:

```bash
python -m venv /tmp/stratfirst-release-venv
/tmp/stratfirst-release-venv/bin/pip install -r requirements.txt
/tmp/stratfirst-release-venv/bin/python main.py test
```

Expected: clean install and green tests without local credentials.

---

## Phase 4 — Give judges a deterministic demo

### 7. Build a one-command offline demo

- [ ] Add `python main.py demo` or `./scripts/demo.sh`.
- [ ] The demo must not require Fyers, Gemini, Tapetide, Tailscale, systemd, or network access.
- [ ] Use deterministic synthetic/sample inputs to demonstrate:
  - multiple strategy intents;
  - rules allocation;
  - with-LLM versus no-LLM counterfactual weights;
  - simulated fills and safety rejections;
  - LightGBM shadow output or a clearly labelled sample;
  - generated dashboard/report.
- [ ] Never imply that synthetic demo P&L is investment evidence.
- [ ] Make repeat runs idempotent and write only to a disposable demo output directory.
- [ ] Add an integration test for the demo command.

Acceptance:

```bash
.venv/bin/python main.py demo
test -f data/demo/dashboard.html
.venv/bin/python -m pytest tests/test_demo.py -q
```

Expected: one command completes offline and prints where to open the report.

### 8. Prepare public sample evidence

- [ ] Create `docs/results/RESULTS.md`.
- [ ] Export only small, redacted, reproducible summaries under `docs/results/`.
- [ ] Include:
  - training window, rows, dates, folds, embargo;
  - mean AUC and top-five precision;
  - model, rules, equal/random, and Nifty comparisons;
  - forward sample size prominently;
  - with-LLM/no-LLM weight example;
  - execution assumptions and known limitations.
- [ ] Add one dashboard screenshot with no private paths, IPs, or account data.
- [ ] Link these artifacts from `README.md`.
- [ ] Reconcile stale numbers in `docs/paper/limitations-and-methods.md` with the current manifest (`mean_auc` and `mean_top5_precision` currently disagree).

Acceptance:

```bash
rg -n '0\.7395|0\.4760|12363' README.md docs
```

Expected: stale metrics removed or explicitly tied to a dated older run.

---

## Phase 5 — Finish judge-facing documentation

### 9. Review the pitch and architecture

- [ ] Verify the README says 21 implemented and 13 currently enabled.
- [ ] Verify it never says the strategies are “proven profitable.”
- [ ] Verify Gemini is described as meta-only and LightGBM as shadow-only.
- [ ] Render the Mermaid diagram on GitHub or replace it with an exported SVG.
- [ ] Add links to the public results screenshot/artifacts.
- [ ] Confirm the quick-start commands work from a clean clone.

### 10. Preserve the failure-recovery story

- [ ] Validate every incident in `docs/FAILURE_RECOVERY.md` against code/tests.
- [ ] Choose the websocket close-window incident as the primary application answer.
- [ ] Keep the answer concise: symptom → hidden cause → complete fix → lesson.
- [ ] Use model underperformance and shadow gating as the secondary AI-judgment story.
- [ ] Remove implementation history that cannot be substantiated.

### 11. Add a submission page

- [ ] Add `SUBMISSION.md` containing:
  - project name;
  - 100–150 word problem statement;
  - 100–150 word solution/pitch;
  - architecture summary;
  - measured evidence;
  - “what broke and how we recovered” answer;
  - exact demo commands;
  - public repository and pitch-video placeholders.
- [ ] Keep application copy consistent with `README.md`.

---

## Phase 6 — Add public project hygiene

### 12. Add a license

- [ ] Ask the user to choose a license. Recommend MIT for a permissive student project unless they want restrictions.
- [ ] Add the chosen root `LICENSE`.
- [ ] Confirm third-party datasets/models are not being relicensed improperly.

### 13. Add CI

- [ ] Add `.github/workflows/test.yml`.
- [ ] Use a supported Python version matching local development.
- [ ] Install from `requirements.txt` and run `python main.py test`.
- [ ] Ensure CI does not require secrets, broker access, systemd, or live network calls.
- [ ] Add the CI badge only after the first public run succeeds.

Acceptance:

```bash
.venv/bin/python main.py test
```

Expected: green locally and on GitHub Actions.

---

## Phase 7 — Create the clean public repository

### 14. Review the exact first-commit boundary

- [ ] Initialize Git only after the user explicitly approves.
- [ ] Stage source, tests, config templates, deployment templates, public docs, sanitized examples, and small result summaries.
- [ ] Do not stage `.env`, runtime data, logs, caches, models with unclear redistribution rights, market history, or internal agent planning files.
- [ ] Review every staged path and staged diff.

Acceptance:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

- [ ] Confirm repository size is reasonable:

```bash
git ls-files -z | xargs -0 du -ch | tail -1
```

### 15. Commit and publish only on explicit request

- [ ] Ask the user for GitHub owner, repository visibility, and final name.
- [ ] Recommend repository name `stratfirst-ai`.
- [ ] Create the commit only when explicitly asked.
- [ ] Create/push the public GitHub repository only when explicitly asked.
- [ ] Confirm the repository can be cloned into a fresh directory and the judge-safe demo works.

---

## Phase 8 — Final Buildathon package

### 16. Record the five-minute pitch

- [ ] 0:00–0:35 — problem: opaque AI trading and unsafe deployment.
- [ ] 0:35–1:10 — strat-first thesis and 21/13 selection discipline.
- [ ] 1:10–2:00 — architecture and AI boundaries.
- [ ] 2:00–3:15 — one-command demo, dashboard, and counterfactual LLM weights.
- [ ] 3:15–4:05 — measured results, including benchmark underperformance and shadow gating.
- [ ] 4:05–4:40 — websocket failure/recovery story.
- [ ] 4:40–5:00 — value: a novice can run a disciplined experiment without unproven AI touching money.
- [ ] Upload as an unlisted video and test playback without authentication.

### 17. Final release gate

- [ ] Public repository opens without authentication.
- [ ] README renders correctly.
- [ ] Mermaid/SVG and screenshot render.
- [ ] Clean-clone demo works.
- [ ] CI is green.
- [ ] No secrets/private paths are present.
- [ ] Results include dates, sample sizes, assumptions, and limitations.
- [ ] Application claims match current artifacts.
- [ ] Pitch video is under five minutes and accessible.

Final commands:

```bash
python main.py test
python main.py diagnose --json
python main.py demo
git status --short
git grep -n '/home/loki\|TAPETIDE_TOKEN=.*[^=]\|FYERS_ACCESS_TOKEN=.*[^=]\|GEMINI_API_KEY=.*[^=]'
```

Stop and report any failure. Do not publish around a failed gate.

