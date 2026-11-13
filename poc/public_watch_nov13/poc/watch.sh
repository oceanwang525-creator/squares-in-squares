#!/bin/zsh
# Periodic public-source watcher: notebooks + datasets, certified per-n wins into state/best.json.
# Run from a scratch working directory with the repo's virtualenv active:
#   LIT=<literal report json> PD=<pandas report json> /path/to/watch.sh [cycles]
# The two reports list the certified L per n of the incumbent ({"results": [{"n": .., "L": ".."}]}).
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
B="$ROOT/poc/pell_corrugation/data/border-scan"
LIT="${LIT:-$B/merged-literal-official.json}"; PD="${PD:-$B/merged-pandas-official.json}"
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
for i in $(seq 1 ${1:-12}); do
  echo "=== cycle $i $(date '+%H:%M')" >> watch.log
  python "$HERE/listk.py" > /dev/null 2>> watch.log
  python -u "$HERE/merge_public.py" "$B/merged.csv" "$LIT" "$PD" kernels.json 2>&1 | grep -E "win n=|DONE" >> watch.log
  python "$HERE/listd.py" >> watch.log 2>&1
  python -u "$HERE/merge_datasets.py" "$B/merged.csv" "$LIT" "$PD" datasets.json 2>&1 | grep -E "win n=|DONE" >> watch.log
  sleep 2400
done
