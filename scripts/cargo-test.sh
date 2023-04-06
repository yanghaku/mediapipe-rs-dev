#!/bin/bash

set -ex

source "$HOME/.cargo/env"

CURRENT="$(realpath "$(dirname -- "$0")")"
WASI_SYSROOT="${CURRENT}/../assets/wasi-sysroot"
INCLUDES="-I${WASI_SYSROOT}/c++/v1 -I${WASI_SYSROOT}/bits -I${WASI_SYSROOT}/wasi"
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=${WASI_SYSROOT} --target=wasm32-wasi -fvisibility=default ${INCLUDES}"

# default features (audio,text,vision)
cargo test --release -- --nocapture

# ffmpeg features
# 1. audio
cargo test --test audio_classification --release --no-default-features --features="ffmpeg" -- --nocapture
