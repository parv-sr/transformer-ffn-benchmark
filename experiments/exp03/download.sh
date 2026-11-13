#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/exp03/{forest,mushrooms,susy}
kaggle datasets download -d uciml/forest-cover-type-dataset -p data/exp03/forest --unzip
kaggle competitions download -c playground-series-s4e8 -p data/exp03/mushrooms
unzip -o -q data/exp03/mushrooms/playground-series-s4e8.zip -d data/exp03/mushrooms
rm -f data/exp03/mushrooms/playground-series-s4e8.zip
kaggle datasets download -d janus137/supersymmetry-dataset -p data/exp03/susy --unzip
python -m experiments.exp03.exp03 --dry-run
