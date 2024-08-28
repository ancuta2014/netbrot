#!/usr/bin/bash

# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

function with_echo() {
    echo "+++" "$@"
    "$@"
}

infile=${1}
index_start=${2:-0}
index_end=${3:-${index_start}}
suffix=$(basename "${infile}" .npz)

for i in $(seq -f '%02g' ${index_start} ${index_end}); do
    with_echo python scripts/generate-matrix-gallery.py \
        --overwrite --quiet \
        --max-escape-radius 40 \
        --ranges "${i}" \
        --suffix "${suffix}" \
        --outfile 'src/gallery.rs' \
        --infile "${infile}"
    with_echo cargo fmt -- 'src/gallery.rs'
    sed -i "64s/gallery::EXHIBIT_.*/gallery::EXHIBIT_${i}_${suffix^^};/" 'src/main.rs'

    with_echo cargo build --release
    # with_echo ./target/release/netbrot -r 1200 -- "result-${suffix,,}-${i}.png"
done
