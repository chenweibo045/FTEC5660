#!/usr/bin/env bash
# Download the public_test receipt images from the course template repo.
# Usage: bash fetch_public_test.sh
set -euo pipefail

BASE="https://raw.githubusercontent.com/HieuNT91/FTEC5660/main/public_test"
mkdir -p public_test

for n in 1 2 3 4 5 6 7; do
    curl -fSL "$BASE/receipt${n}.jpg" -o "public_test/receipt${n}.jpg"
done
curl -fSL "$BASE/ground_truth.json" -o "public_test/ground_truth.json"

echo "public_test/ ready: $(ls public_test | wc -l | tr -d ' ') files"
