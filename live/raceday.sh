#!/usr/bin/env bash
# Race-day helper for laptop live mode (Git Bash).
#   live/raceday.sh login                                   sign in to F1 TV once (re-run on race day)
#   live/raceday.sh refresh                                 run the post-qualifying Databricks job (~25 min), then pull
#   live/raceday.sh pull                                    copy the latest site data from Databricks and rebuild the site
#   live/raceday.sh test                                    replay the 2025 British GP (alert around lap 44)
#   live/raceday.sh live YEAR ROUND "RACE NAME" LAPS [--later]
set -euo pipefail
cd "$(dirname "$0")/.."
# started from cmd, `bash` is WSL: re-run under Git Bash so the Windows venv, CLI and npm are used
GITBASH="/mnt/c/Program Files/Git/bin/bash.exe"
[ -z "${MSYSTEM:-}" ] && [ -x "$GITBASH" ] && exec "$GITBASH" live/raceday.sh "$@"

PY=live/.venv/Scripts/python
DB="${DATABRICKS_CLI:-$LOCALAPPDATA/Microsoft/WinGet/Links/databricks.exe}"
GRID_JOB=407815354068899
export MSYS_NO_PATHCONV=1
# F1 account for FastF1 sign-in: live/.keys/f1.env (gitignored) with F1_USERNAME=... and F1_PASSWORD=...
[ -f live/.keys/f1.env ] && { set -a; . live/.keys/f1.env; set +a; }

pull() {
  "$DB" fs cp -r dbfs:/Volumes/workspace/default/f1_data/site_data site/public/data --overwrite > /dev/null
  test -f site/public/data/index.json
  (cd site && npm run build)
}

case "${1:-}" in
  login)   "$PY" -c "from fastf1.internals.f1account import get_token; get_token(); print('Signed in to F1 TV.')" ;;
  refresh) "$DB" jobs run-now "$GRID_JOB" && pull ;;
  pull)    pull ;;
  test)    "$PY" -m live.serve --replay live/out/2025-12.csv --slug 2025-12 --every "${2:-2}" ;;
  live)
    [ $# -ge 5 ] || { echo 'usage: live/raceday.sh live YEAR ROUND "RACE NAME" LAPS [--later]'; exit 1; }
    "$PY" -m live.serve --live --year "$2" --round "$3" --race "$4" --laps "$5" "${@:6}" ;;
  *) sed -n 2,7p "$0"; exit 1 ;;
esac
