#!/usr/bin/env bash
# Export redacted glance and push only docs/site/ so GitHub Pages stays current.
# Safe for systemd oneshots: no prompts; no-op if nothing changed / not a git checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new}"

# User systemd oneshots often lack SSH_AUTH_SOCK; pull it from the user manager env.
if [[ -z "${SSH_AUTH_SOCK:-}" ]] && command -v systemctl >/dev/null 2>&1; then
  _sock="$(systemctl --user show-environment 2>/dev/null | sed -n 's/^SSH_AUTH_SOCK=//p' || true)"
  if [[ -n "${_sock}" && -S "${_sock}" ]]; then
    export SSH_AUTH_SOCK="${_sock}"
  fi
fi
if [[ -z "${SSH_AUTH_SOCK:-}" && -d "${HOME}/.ssh/agent" ]]; then
  _sock="$(find "${HOME}/.ssh/agent" -type s -name 's.*.agent.*' 2>/dev/null | head -n1 || true)"
  if [[ -n "${_sock}" ]]; then
    export SSH_AUTH_SOCK="${_sock}"
  fi
fi

if [[ "${NSE_TRADER_SKIP_PAGES_PUBLISH:-}" =~ ^(1|true|yes)$ ]]; then
  echo "Skipping Pages publish (NSE_TRADER_SKIP_PAGES_PUBLISH set)."
  exit 0
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

if [[ ! -d "$ROOT/.git" ]]; then
  echo "Not a git checkout — skip Pages publish."
  exit 0
fi

if ! git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  echo "No upstream branch — skip Pages publish."
  exit 0
fi

"$ROOT/.venv/bin/python" main.py public-glance
git add docs/site/glance.json docs/site/index.html docs/site/.nojekyll
if git diff --cached --quiet; then
  echo "No public-glance changes to publish."
  exit 0
fi

# Refuse if anything outside docs/site is staged.
while IFS= read -r f; do
  case "$f" in
    docs/site/*) ;;
    *)
      echo "Refusing to publish unexpected path: $f" >&2
      git reset HEAD --quiet
      exit 1
      ;;
  esac
done < <(git diff --cached --name-only)

git commit -m "Refresh public GitHub Pages glance (redacted)."
git push origin HEAD
echo "Published. Pages will update after the pages workflow finishes."
