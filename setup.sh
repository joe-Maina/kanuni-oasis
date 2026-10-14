#!/usr/bin/env bash
# Fetch OASIS and AgentSociety 2 at the pinned upstream commits into the
# folders our code expects, then create one Python 3.11 venv for each.
# Upstream code is not modified; ours lives next to it (kanuni/, kanuni_as2/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OASIS_COMMIT=dded86a1121d52ecd5b9320f40326272cfc5fa42   # camel-ai/oasis
AS2_COMMIT=8cc5bb9a9c27d039ec5519375db9d73ccf508cba     # tsinghua-fib-lab/AgentSociety, release 2.9.3
PYTHON="${PYTHON:-python3.11}"

fetch() {  # repo commit dest
  local repo=$1 commit=$2 dest=$3
  if [ -f "$dest/pyproject.toml" ]; then
    echo "$repo already present in $dest"
    return
  fi
  echo "Fetching $repo@$commit into $dest"
  mkdir -p "$dest"
  curl -fsSL "https://codeload.github.com/$repo/tar.gz/$commit" |
    tar xz --strip-components=1 -C "$dest"
}

fetch camel-ai/oasis "$OASIS_COMMIT" "$ROOT/camel-oasis/oasis"
fetch tsinghua-fib-lab/AgentSociety "$AS2_COMMIT" "$ROOT/agentic-society/AgentSociety"

"$PYTHON" -m venv "$ROOT/camel-oasis/oasis/.venv"
"$ROOT/camel-oasis/oasis/.venv/bin/pip" install -q -e "$ROOT/camel-oasis/oasis"

"$PYTHON" -m venv "$ROOT/agentic-society/.venv"
"$ROOT/agentic-society/.venv/bin/pip" install -q \
  -e "$ROOT/agentic-society/AgentSociety/packages/agentsociety2"

cat <<EOF
Done. Next:
  1. Create camel-oasis/oasis/kanuni/.env with the model endpoint
     (see camel-oasis/oasis/kanuni/README.md).
  2. Start the shim:  cd camel-oasis/oasis && .venv/bin/python -m kanuni.ollama_shim --port 8765
EOF
