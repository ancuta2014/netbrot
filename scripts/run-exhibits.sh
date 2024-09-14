#!/usr/bin/bash

# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

function with_echo() {
    echo "+++" "$@"
    # "$@"
}

suffix=$(date "+%Y%m%d-%H%M%S")

for filename in $@; do
    with_echo ./target/release/netbrot \
        --resolution 1200 \
        --maxit 512 \
        --outfile "${filename%.json}-${suffix}.png" \
        "${filename}"
done
