#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
clone_pin () {
  local url="$1" name="$2" sha="$3"
  if [ ! -d "$ROOT/third_party/$name/.git" ]; then
    git clone "$url" "$ROOT/third_party/$name"
  fi
  git -C "$ROOT/third_party/$name" fetch --all --tags
  git -C "$ROOT/third_party/$name" checkout "$sha"
}
clone_pin https://github.com/qiqi-lu/fluoresfm.git fluoresfm 772ac54b2090d4de343979b8ba221c4de541ff85
clone_pin https://github.com/Feynben/UniMIE.git UniMIE b46975d7c6f8c2847b382f1d1e801a1060aa40a8
clone_pin https://github.com/JonghoLee0/CASPI.git CASPI 0fe5024e6221f1470f8d4e3e48b346dd38a6fb12
printf '\nPinned third-party repositories are ready.\n'
