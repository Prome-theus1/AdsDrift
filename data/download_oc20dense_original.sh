#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/path/to/project_OC2M/data
RAW_ROOT="$DATA_ROOT/oc20_dense_original"
ARCHIVE_ROOT="$RAW_ROOT/archives"
EXTRACT_ROOT="$RAW_ROOT/extracted"
LOG_ROOT="$RAW_ROOT/logs"

mkdir -p "$ARCHIVE_ROOT" "$EXTRACT_ROOT" "$LOG_ROOT"

download() {
    local url=$1
    local output=$2
    curl --fail --location --retry 12 --retry-delay 10 \
        --continue-at - --output "$output" "$url"
}

download \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_data.tar.gz \
    "$ARCHIVE_ROOT/oc20_dense_data.tar.gz"
echo "0163b0e8c4df6d9c426b875a28d9178a  $ARCHIVE_ROOT/oc20_dense_data.tar.gz" \
    | md5sum --check -

download \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_mappings.tar.gz \
    "$ARCHIVE_ROOT/oc20_dense_mappings.tar.gz"
MAPPING_ARCHIVE_MD5=$(md5sum "$ARCHIVE_ROOT/oc20_dense_mappings.tar.gz" | awk '{print $1}')
printf '%s  %s\n' "$MAPPING_ARCHIVE_MD5" \
    "$ARCHIVE_ROOT/oc20_dense_mappings.tar.gz" \
    > "$ARCHIVE_ROOT/oc20_dense_mappings.tar.gz.md5"
if [[ "$MAPPING_ARCHIVE_MD5" != "c18735c405ce6ce5761432b07287d8d9" ]]; then
    printf '%s\n' \
        "WARNING: mapping archive wrapper MD5 differs from the value published by FAIR-Chem." \
        "The extracted mapping payloads will be checked individually before acceptance." \
        >&2
fi

download \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/large_files/bulks.pkl \
    "$ARCHIVE_ROOT/bulks.pkl"
sha256sum "$ARCHIVE_ROOT/bulks.pkl" > "$ARCHIVE_ROOT/bulks.pkl.sha256"

mkdir -p "$EXTRACT_ROOT/lmdb" "$EXTRACT_ROOT/mappings"
tar --extract --gzip --file "$ARCHIVE_ROOT/oc20_dense_data.tar.gz" \
    --directory "$EXTRACT_ROOT/lmdb"
tar --extract --gzip --file "$ARCHIVE_ROOT/oc20_dense_mappings.tar.gz" \
    --directory "$EXTRACT_ROOT/mappings"

# FAIR-Chem publishes checksums for each mapping payload. These content-level
# checks are stronger than relying only on the gzip wrapper checksum, which can
# change when an otherwise identical tarball is recompressed.
cat > "$ARCHIVE_ROOT/oc20dense_mapping_payloads.md5" <<'EOF'
3e26c3bcef01ccfc9b001931065ea6e6  oc20dense_mapping.pkl
fd589b013b72e62e11a6b2a5bd1d323c  oc20dense_targets.pkl
78d25997e0aaf754df526ab37276bb89  oc20dense_compute.pkl
b07c64158e4bfa5f7b9bf6263753ecc5  oc20dense_ref_energies.pkl
1ba0bc266130f186850f5faa547b6a02  oc20dense_tags.pkl
EOF
(
    cd "$EXTRACT_ROOT/mappings"
    md5sum --check "$ARCHIVE_ROOT/oc20dense_mapping_payloads.md5"
)

find "$EXTRACT_ROOT" -type f -print | sort > "$RAW_ROOT/extracted_files.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RAW_ROOT/_DOWNLOAD_AND_EXTRACT_SUCCESS"
