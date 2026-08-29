#!/usr/bin/env bash
# Export redacted glance and push only docs/site/ so GitHub Pages stays current.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" main.py public-glance
git add docs/site/glance.json docs/site/index.html docs/site/.nojekyll
if git diff --cached --quiet; then
  echo "No public-glance changes to publish."
  exit 0
fi
# Refuse if anything outside docs/site is staged.
mapfile -t staged < <(git diff --cached --name-only)
for f in "${staged[@]}"; do
  case "$f" in
    docs/site/*) ;;
    *)
      echo "Refusing to publish unexpected path: $f" >&2
      exit 1
      ;;
  esac
done

git commit -m "Refresh public GitHub Pages glance (redacted)."
git push origin HEAD
echo "Published. Pages will update after the pages workflow finishes."
