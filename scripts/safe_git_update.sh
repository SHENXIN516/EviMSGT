#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
BRANCH="${2:-main}"
BACKUP_DIR="${EVIMSGT_SLURM_LOG_BACKUP:-$HOME/EviMSGT_slurm_logs}"

if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
  echo "A rebase is already in progress. Resolve it first, then rerun this script."
  echo "Useful commands:"
  echo "  git status"
  echo "  git rebase --continue"
  echo "  git rebase --abort"
  exit 1
fi

mkdir -p "$BACKUP_DIR"

echo "Moving local Slurm logs to: $BACKUP_DIR"
find . -maxdepth 1 -type f \( -name 'slurm-*.out' -o -name 'slurm-*.err' \) -print0 |
  while IFS= read -r -d '' path; do
    base="$(basename "$path")"
    if [ -e "$BACKUP_DIR/$base" ]; then
      mv "$path" "$BACKUP_DIR/${base}.$(date +%Y%m%d%H%M%S)"
    else
      mv "$path" "$BACKUP_DIR/$base"
    fi
  done

echo "Stashing local non-log changes, if any."
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -m "safe_git_update local changes"
else
  echo "No tracked local changes to stash."
fi

echo "Fetching $REMOTE/$BRANCH"
git fetch "$REMOTE" "$BRANCH"

echo "Rebasing local commits on $REMOTE/$BRANCH"
git rebase "$REMOTE/$BRANCH"

echo "Update finished."
echo "If a stash was created, restore it with:"
echo "  git stash list"
echo "  git stash pop"
