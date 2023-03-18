#!/bin/bash

# install curl
apt update && apt install curl -y

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# add wasm32-wasi target
rustup target add wasm32-wasi
