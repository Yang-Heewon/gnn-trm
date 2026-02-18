#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/00_preprocess.sh"
bash "$SCRIPT_DIR/01_embed.sh"
bash "$SCRIPT_DIR/02_train_phase1.sh"
bash "$SCRIPT_DIR/03_train_phase2_rl.sh"
