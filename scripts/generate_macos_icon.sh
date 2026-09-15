#!/usr/bin/env bash

set -euo pipefail

seeg_script_dir=$(dirname "$0")
seeg_script_dir=$(cd "$seeg_script_dir"; pwd -P)
seeg_repository_root=$(cd "$seeg_script_dir/.."; pwd -P)
seeg_source_icon="$seeg_repository_root/packaging/macos/seeg-localizer-icon.png"
seeg_output_icon="$seeg_repository_root/packaging/macos/seeg-localizer.icns"

if [[ $(uname -s) != "Darwin" ]]; then
    echo "macOS icon generation requires the macOS sips and iconutil tools." >&2
    exit 1
fi

if [[ ! -f "$seeg_source_icon" ]]; then
    echo "Icon source not found: $seeg_source_icon" >&2
    exit 1
fi

seeg_icon_work_dir=$(mktemp -d "${TMPDIR:-/tmp}/seeg-localizer-icon.XXXXXX")
seeg_iconset="$seeg_icon_work_dir/sEEG-Localizer.iconset"
mkdir "$seeg_iconset"

cleanup() {
    if [[ -d "$seeg_icon_work_dir" ]]; then
        rm -R "$seeg_icon_work_dir"
    fi
}
trap cleanup EXIT

make_icon() {
    local size=$1
    local filename=$2
    sips -z "$size" "$size" "$seeg_source_icon" \
        --out "$seeg_iconset/$filename" >/dev/null
}

make_icon 16 icon_16x16.png
make_icon 32 icon_16x16@2x.png
make_icon 32 icon_32x32.png
make_icon 64 icon_32x32@2x.png
make_icon 128 icon_128x128.png
make_icon 256 icon_128x128@2x.png
make_icon 256 icon_256x256.png
make_icon 512 icon_256x256@2x.png
make_icon 512 icon_512x512.png
make_icon 1024 icon_512x512@2x.png

iconutil -c icns "$seeg_iconset" -o "$seeg_output_icon"
echo "Created $seeg_output_icon"
