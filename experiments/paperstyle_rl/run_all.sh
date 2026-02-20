#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/00_download.sh"
bash "$SCRIPT_DIR/01_embed.sh"
bash "$SCRIPT_DIR/02_train_phase1.sh"
bash "$SCRIPT_DIR/03_train_phase2_rl.sh"
bash "$SCRIPT_DIR/04_eval_phase2_test.sh"
