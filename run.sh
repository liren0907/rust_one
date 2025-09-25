#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory of this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ensure_bin() {
  echo "Building ort-tutorial (release)..."
  RUSTFLAGS="-C target-cpu=apple-m1 -C llvm-args=-unroll-threshold=1000" \
    cargo build --release -p ort-tutorial
}

main() {
  ensure_bin
  export DYLD_LIBRARY_PATH="$SCRIPT_DIR/target/release:${DYLD_LIBRARY_PATH:-}"

  # Run the simplified ort-tutorial demonstration
  cargo run -p ort-tutorial

}

main "$@"
