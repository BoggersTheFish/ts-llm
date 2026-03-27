#!/usr/bin/env bash
# Low CPU / low RAM git push (avoids zlib pegging weak laptops during pack-objects).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
exec git \
  -c pack.compression=0 \
  -c pack.threads=1 \
  -c core.preloadindex=false \
  push "$@"
