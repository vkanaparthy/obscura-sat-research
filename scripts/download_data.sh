#!/usr/bin/env bash
# Download and unzip the OBSCURA competition data into data/raw/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle competitions download -c obscura -p data/raw
unzip -q -o data/raw/obscura.zip -d data/raw
rm data/raw/obscura.zip
ls data/raw | head
