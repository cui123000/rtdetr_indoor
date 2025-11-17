#!/usr/bin/env bash
# Download COCO 2017 dataset (train/val and annotations) with resume and extraction.
# Also optionally download label zips (box labels and/or segmentation labels).
# Usage:
#   ./download_coco2017.sh [--dst DIR] [--with-test] [--with-labels] [--with-segments] [--skip-extract]
# Examples:
#   ./download_coco2017.sh                              # to datasets/coco under repo root
#   ./download_coco2017.sh --dst /data/coco             # custom destination
#   ./download_coco2017.sh --with-test                  # also download test2017.zip
#   ./download_coco2017.sh --with-labels                # download box labels zip
#   ./download_coco2017.sh --with-segments              # download segmentation labels zip
#   ./download_coco2017.sh --with-labels --with-segments
#   ./download_coco2017.sh --skip-extract               # only download zips

set -euo pipefail

# Resolve repo root based on this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DST="${REPO_ROOT}/datasets/coco"
WITH_TEST=0
WITH_LABELS=0
WITH_SEGMENTS=0
SKIP_EXTRACT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst)
      DST="$2"
      shift 2
      ;;
    --with-test)
      WITH_TEST=1
      shift
      ;;
    --with-labels)
      WITH_LABELS=1
      shift
      ;;
    --with-segments)
      WITH_SEGMENTS=1
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${DST}" && cd "${DST}"

echo "Destination: ${DST}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

_download() {
  local url="$1"; local out="$2";
  if [[ -f "$out" ]]; then
    echo "Already exists: $out (skip)"
    return 0
  fi
  echo "Downloading: $url -> $out"
  if have_cmd wget; then
    wget -c -O "$out" "$url"
  elif have_cmd curl; then
    curl -L -C - -o "$out" "$url"
  else
    echo "Neither wget nor curl is available." >&2
    exit 1
  fi
}

TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
TEST_URL="http://images.cocodataset.org/zips/test2017.zip"
ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LABELS_URL_BASE="https://github.com/ultralytics/assets/releases/download/v0.0.0"

# Download
_download "$TRAIN_URL" "train2017.zip"
_download "$VAL_URL"   "val2017.zip"
_download "$ANNO_URL"  "annotations_trainval2017.zip"
if [[ "$WITH_TEST" -eq 1 ]]; then
  _download "$TEST_URL"  "test2017.zip"
fi

echo "All downloads finished."

if [[ "$SKIP_EXTRACT" -eq 1 ]]; then
  echo "Skip extraction as requested."
  exit 0
fi

# Extract
extract_zip() {
  local zipf="$1"
  if [[ ! -f "$zipf" ]]; then
    echo "Missing zip: $zipf (skip)"
    return 0
  fi
  echo "Extracting: $zipf -> $(dirname "$zipf")"
  if have_cmd unzip; then
    unzip -q -o "$zipf" -d "$(dirname "$zipf")"
  elif have_cmd bsdtar; then
    bsdtar -xf "$zipf" -C "$(dirname "$zipf")"
  else
    echo "Neither unzip nor bsdtar is available. Please install unzip." >&2
    exit 1
  fi
}

extract_zip train2017.zip
extract_zip val2017.zip
extract_zip annotations_trainval2017.zip
if [[ "$WITH_TEST" -eq 1 ]]; then
  extract_zip test2017.zip
fi

# Optional: download label zips (box labels / segments) into datasets parent dir
if [[ "$WITH_LABELS" -eq 1 || "$WITH_SEGMENTS" -eq 1 ]]; then
  LABELS_DIR="$(dirname "${DST}")"
  mkdir -p "${LABELS_DIR}"
  if [[ "$WITH_LABELS" -eq 1 ]]; then
    _download "${LABELS_URL_BASE}/coco2017labels.zip" "${LABELS_DIR}/coco2017labels.zip"
    extract_zip "${LABELS_DIR}/coco2017labels.zip"
  fi
  if [[ "$WITH_SEGMENTS" -eq 1 ]]; then
    _download "${LABELS_URL_BASE}/coco2017labels-segments.zip" "${LABELS_DIR}/coco2017labels-segments.zip"
    extract_zip "${LABELS_DIR}/coco2017labels-segments.zip"
  fi
fi

echo "Extraction done. Structure:"
ls -lah --group-directories-first || true

echo "COCO 2017 is ready at: ${DST}"
