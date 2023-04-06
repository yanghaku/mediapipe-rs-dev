#!/bin/bash

set -ex

source "$HOME/.cargo/env"

CURRENT="$(dirname -- "$0")"
SYSROOT="${CURRENT}/../assets/wasi-sysroot"
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=${SYSROOT} --target=wasm32-wasi -fvisibility=default"

# default features (audio,text,vision)
cargo test --release -- --nocapture

# ffmpeg features
# 1. audio
cargo test --test audio_classification --release --no-default-features --features="ffmpeg" -- --nocapture
