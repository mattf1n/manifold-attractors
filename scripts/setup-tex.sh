#!/usr/bin/env bash
# Rootless Linux x86_64 reconstruction environment; NOT the lost original setup.
# Usage: bash scripts/setup-tex.sh [installation-prefix]
set -euo pipefail
[[ $(uname -sm) == 'Linux x86_64' ]] || { echo 'Requires Linux x86_64' >&2; exit 1; }
prefix=${1:-"$HOME/.local/texlive/2025"}
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
cache="$prefix/bootstrap"
fonts="$HOME/.local/share/fonts/manifold-attractors"
repo=https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final
mkdir -p "$cache" "$fonts"
fetch() {
  local url=$1 output=$2 hash=$3
  if ! echo "$hash  $output" | sha256sum --check --status 2>/dev/null; then
    curl --fail --location --retry 3 "$url" -o "$output"
  fi
  echo "$hash  $output" | sha256sum --check
}
fetch "$repo/install-tl-unx.tar.gz" "$cache/install-tl-unx.tar.gz" \
  311df9f1477fd90c520159d1feddc2d6270f010d8349d1f6bdb9461a93b48a5c
# Frozen repository identity. tlmgr also verifies package hashes from this DB.
# Its historic signing key is expired; this pinned digest is intentional.
fetch "$repo/tlpkg/texlive.tlpdb.xz" "$cache/texlive.tlpdb.xz" \
  56df038e9070a3d587eeceb5def3f154b2cacccef36d4295bb49e0618977c34a
if [[ ! -x "$prefix/bin/x86_64-linux/tlmgr" ]]; then
  tar -xzf "$cache/install-tl-unx.tar.gz" -C "$cache"
  printf '%s\n' \
    'selected_scheme scheme-basic' "TEXDIR $prefix" \
    "TEXMFCONFIG $prefix/texmf-user-config" "TEXMFHOME $prefix/texmf-home" \
    "TEXMFLOCAL $prefix/texmf-local" "TEXMFSYSCONFIG $prefix/texmf-config" \
    "TEXMFSYSVAR $prefix/texmf-var" "TEXMFVAR $prefix/texmf-user-var" \
    'binary_x86_64-linux 1' 'instopt_adjustpath 0' 'instopt_letter 0' \
    'instopt_portable 0' 'instopt_write18_restricted 1' \
    'tlpdbopt_install_docfiles 0' 'tlpdbopt_install_srcfiles 0' > "$cache/install.profile"
  "$cache/install-tl-20260301/install-tl" -profile "$cache/install.profile" -repository "$repo"
fi
export PATH="$prefix/bin/x86_64-linux:$PATH"
export TEXMFVAR="$prefix/texmf-user-var" TEXMFCONFIG="$prefix/texmf-user-config"
export TEXMFHOME="$prefix/texmf-home"
[[ $(tlmgr --version) == *'version 2025'* ]] || { echo 'Not TeX Live 2025' >&2; exit 1; }
tlmgr --repository "$repo" install \
  luatex luaotfload fontspec newtx txfonts cabin noto inconsolata mathastext \
  siunitx standalone adjustbox pgfplots hobby tempora cm-unicode arphic-ttf \
  unicode-math lualatex-math anyfontsize luatex85 xpatch xstring mweights \
  upquote underscore greek-fontenc cbfonts-fd cbfonts
# TL2025 final has LuaHBTeX 1.22. The actual PDF producer was 1.21.0.
# Overlay ONLY that engine from the TL2025 release-era archive; do not tlmgr update.
fetch https://texlive.info/tlnet-archive/2025/03/09/tlnet/archive/luahbtex.x86_64-linux.tar.xz \
  "$cache/luahbtex-1.21.tar.xz" \
  09adfc8930342b988754b4491e7cb81c0d3e04fbc86753c2953811081cd1d7a6
tar -xf "$cache/luahbtex-1.21.tar.xz" -C "$prefix" bin/x86_64-linux/luahbtex
fmtutil-sys --byfmt lualatex
lualatex --version | head -3
lualatex --version | grep -q 'Version 1.21.0 (TeX Live 2025)'
fetch https://github.com/notofonts/math/releases/download/NotoSansMath-v3.000/NotoSansMath-v3.000.zip \
  "$cache/NotoSansMath-v3.000.zip" \
  ac351837b41f8a897f020b97fb0f075ad574c1e9669fb5839ada1f92fd748356
unzip -p "$cache/NotoSansMath-v3.000.zip" \
  NotoSansMath/googlefonts/ttf/NotoSansMath-Regular.ttf > "$fonts/NotoSansMath-Regular.ttf"
echo "fe380412e05c2cca4b712644c58ed38ed7b1e3153bc483a8b38725fa487b8830  $fonts/NotoSansMath-Regular.ttf" | sha256sum --check
cjk=https://raw.githubusercontent.com/notofonts/noto-cjk/523d033d6cb47f4a80c58a35753646f5c3608a78/Sans/OTF
fetch "$cjk/SimplifiedChinese/NotoSansCJKsc-Regular.otf" "$fonts/NotoSansCJKsc-Regular.otf" \
  2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b
fetch "$cjk/TraditionalChinese/NotoSansCJKtc-Regular.otf" "$fonts/NotoSansCJKtc-Regular.otf" \
  dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4
# Explicitly expose text Noto Sans to fontconfig/luaotfload name lookup.
for face in Regular Bold Italic BoldItalic; do
  ln -sfn "$prefix/texmf-dist/fonts/truetype/google/noto/NotoSans-$face.ttf" "$fonts/NotoSans-$face.ttf"
done
fc-cache -f "$fonts"
luaotfload-tool --update --force
# Record package revisions and hashes; the engine overlay is listed separately.
tlmgr info --only-installed --data name,localrev > "$prefix/packages.tsv"
sha256sum "$prefix/bin/x86_64-linux/luahbtex" "$fonts"/* > "$prefix/font-engine.sha256"
while IFS= read -r -d '' font; do sha256sum "$font"; done < <(
  find "$prefix/texmf-dist/fonts" -type f \( -name '*.otf' -o -name '*.ttf' \) -print0
) >> "$prefix/font-engine.sha256"
printf 'export PATH=%q:"$PATH"\nexport TEXMFVAR=%q\nexport TEXMFCONFIG=%q\nexport TEXMFHOME=%q\n' \
  "$prefix/bin/x86_64-linux" "$TEXMFVAR" "$TEXMFCONFIG" "$TEXMFHOME" > "$prefix/env.sh"
printf '\nActivate: source %q\nReconstructed typography still requires --allow-reconstructed-styles.\n' "$prefix/env.sh"
