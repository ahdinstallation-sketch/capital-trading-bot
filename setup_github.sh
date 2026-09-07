#!/usr/bin/env bash
#
# One-time setup: create a PRIVATE GitHub repo, push the bot, and load the
# credentials from .env into GitHub Secrets.
#
# Secret values are piped from .env straight to `gh secret set` and are never
# printed. Nothing here echoes a credential.

set -euo pipefail
cd "$(dirname "$0")"

REPO_NAME="${1:-capital-trading-bot}"

command -v gh >/dev/null || { echo "gh CLI not found. brew install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run: gh auth login"; exit 1; }
[ -f .env ] || { echo "No .env found."; exit 1; }

# --- read .env without exporting it into the shell's visible environment
get() {
  local key="$1"
  sed -n "s/^${key}=//p" .env | head -1 | tr -d '\r'
}

for required in CAPITAL_API_KEY CAPITAL_IDENTIFIER CAPITAL_PASSWORD; do
  if [ -z "$(get "$required")" ]; then
    echo "ERROR: $required is empty in .env"; exit 1
  fi
done

# --- git
if [ ! -d .git ]; then
  git init -q
  git branch -M main
fi

git add -A
git diff --cached --quiet || git commit -q -m "Capital.com trading bot"

if ! gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  echo "Creating private repo: $REPO_NAME"
  gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
else
  echo "Repo $REPO_NAME already exists, pushing"
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "$(gh repo view "$REPO_NAME" --json sshUrl -q .sshUrl)"
  git push -u origin main
fi

# --- secrets (values piped in, never echoed)
echo "Setting secrets..."
for key in CAPITAL_API_KEY CAPITAL_IDENTIFIER CAPITAL_PASSWORD; do
  printf '%s' "$(get "$key")" | gh secret set "$key" --repo "$REPO_NAME"
  echo "  set $key"
done

# Non-secret config, editable from the GitHub UI later.
gh variable set DRY_RUN --body "true" --repo "$REPO_NAME" >/dev/null
gh variable set CAPITAL_EPIC --body "$(get CAPITAL_EPIC)" --repo "$REPO_NAME" >/dev/null
echo "  set DRY_RUN=true and CAPITAL_EPIC"

# --- confirm .env did not get committed
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo ""
  echo "*** WARNING: .env IS TRACKED BY GIT. Remove it before pushing further:"
  echo "***   git rm --cached .env && git commit -m 'untrack .env'"
  exit 1
fi

cat <<'EOF'

Done. The workflow runs every 30 minutes in DRY RUN mode.

It will NOT place orders until you arm it, which is one command:

  gh secret set CAPITAL_I_UNDERSTAND_THIS_IS_REAL_MONEY --body "yes"
  gh variable set DRY_RUN --body "false"

Watch it:            gh run list --workflow=trade.yml
Trigger one now:     gh workflow run trade.yml
Read a run's output: gh run view --log

EOF
