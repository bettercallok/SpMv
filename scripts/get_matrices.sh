#!/usr/bin/env bash
# Downloads the Bell & Garland SuiteSparse set into matrices/ (or $1), skipping ones already there.
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/matrices}"
mkdir -p "$DEST"
for entry in Williams/cant Boeing/pwtk DNVS/shipsec1 Williams/consph Williams/pdb1HYS Bova/rma10 \
             Williams/cop20k_A Williams/mac_econ_fwd500 Hamm/scircuit Williams/webbase-1M; do
  name="${entry#*/}"
  [[ -f "$DEST/$name.mtx" ]] && { echo "have $name"; continue; }
  echo "downloading $entry"
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/m.tar.gz" "https://sparse.tamu.edu/MM/$entry.tar.gz"
  tar -xzf "$tmp/m.tar.gz" -C "$tmp"
  mv "$tmp/$name/$name.mtx" "$DEST/$name.mtx"
  rm -rf "$tmp"
done
