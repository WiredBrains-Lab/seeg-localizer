#!/usr/bin/env bash

set -euo pipefail

seeg_script_dir=$(dirname "$0")
seeg_script_dir=$(cd "$seeg_script_dir"; pwd -P)
seeg_repository_root=$(cd "$seeg_script_dir/.."; pwd -P)
seeg_python=${SEEG_BUILD_PYTHON:-python3}
seeg_dist_dir="$seeg_repository_root/dist"
seeg_work_dir="$seeg_repository_root/build/pyinstaller"
seeg_app_name="sEEG Localizer"

if [[ $(uname -s) != "Darwin" ]]; then
    echo "The macOS application must be built on macOS." >&2
    exit 1
fi

cd "$seeg_repository_root"

seeg_version=$(
    "$seeg_python" -c \
        'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])'
)
seeg_architecture=$(uname -m)
seeg_dmg_path="$seeg_dist_dir/sEEG-Localizer-$seeg_version-macOS-$seeg_architecture.dmg"

"$seeg_script_dir/generate_macos_icon.sh"

"$seeg_python" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$seeg_dist_dir" \
    --workpath "$seeg_work_dir" \
    "$seeg_repository_root/packaging/macos/seeg_localizer.spec"

seeg_app_path="$seeg_dist_dir/$seeg_app_name.app"
seeg_executable="$seeg_app_path/Contents/MacOS/$seeg_app_name"

if [[ ! -x "$seeg_executable" ]]; then
    echo "Build failed: application executable was not created." >&2
    exit 1
fi

# Import the frozen dependency graph and exercise the command-line parser
# without creating a window or downloading template data.
"$seeg_executable" --help >/dev/null
plutil -lint "$seeg_app_path/Contents/Info.plist" >/dev/null

# Cloud-synced workspaces can add Finder metadata that strict code-signing
# validation rejects even though it is not part of the application itself.
xattr -cr "$seeg_app_path"
codesign --verify --deep --strict "$seeg_app_path"

seeg_staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/seeg-localizer-dmg.XXXXXX")
seeg_mount_dir=$(mktemp -d "${TMPDIR:-/tmp}/seeg-localizer-mount.XXXXXX")
seeg_dmg_mounted=0
cleanup() {
    if [[ $seeg_dmg_mounted -eq 1 ]]; then
        hdiutil detach -quiet "$seeg_mount_dir" || true
    fi
    if [[ -d "$seeg_staging_dir" ]]; then
        rm -R "$seeg_staging_dir"
    fi
    if [[ -d "$seeg_mount_dir" ]]; then
        rm -R "$seeg_mount_dir"
    fi
}
trap cleanup EXIT

seeg_staged_app="$seeg_staging_dir/$seeg_app_name.app"
ditto "$seeg_app_path" "$seeg_staged_app"
xattr -cr "$seeg_staged_app"
codesign --verify --deep --strict "$seeg_staged_app"
ln -s /Applications "$seeg_staging_dir/Applications"

hdiutil create \
    -quiet \
    -volname "$seeg_app_name" \
    -srcfolder "$seeg_staging_dir" \
    -format UDZO \
    -ov \
    "$seeg_dmg_path"

if [[ -n ${SEEG_MACOS_CODESIGN_IDENTITY:-} ]]; then
    codesign \
        --force \
        --timestamp \
        --sign "$SEEG_MACOS_CODESIGN_IDENTITY" \
        "$seeg_dmg_path"
fi

if [[ -n ${SEEG_MACOS_NOTARY_PROFILE:-} ]]; then
    xcrun notarytool submit \
        "$seeg_dmg_path" \
        --keychain-profile "$SEEG_MACOS_NOTARY_PROFILE" \
        --wait
    xcrun stapler staple "$seeg_dmg_path"
fi

hdiutil verify "$seeg_dmg_path" >/dev/null
hdiutil attach \
    -quiet \
    -readonly \
    -nobrowse \
    -mountpoint "$seeg_mount_dir" \
    "$seeg_dmg_path"
seeg_dmg_mounted=1
codesign --verify --deep --strict "$seeg_mount_dir/$seeg_app_name.app"
hdiutil detach -quiet "$seeg_mount_dir"
seeg_dmg_mounted=0

seeg_dmg_filename=$(basename "$seeg_dmg_path")
(
    cd "$seeg_dist_dir"
    shasum -a 256 "$seeg_dmg_filename" > "$seeg_dmg_filename.sha256"
)

echo "Created $seeg_dmg_path"
