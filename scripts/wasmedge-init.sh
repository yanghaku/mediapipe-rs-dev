#!/bin/bash

# Init the Wasmedge environment  (with wasi-nn plugin and tf-lite backend)

set -ex

source "$(dirname -- "$0")/env.sh"

export WASMEDGE_VERSION=0.12.0-alpha.2

wasmedge_with_nn_init() {
  curl -sLO https://github.com/WasmEdge/WasmEdge/releases/download/${WASMEDGE_VERSION}/WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
  curl -sLO https://github.com/WasmEdge/WasmEdge/releases/download/${WASMEDGE_VERSION}/WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  tar -zxf WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
  tar -zxf WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  mkdir -p "${WASMEDGE_BIN_PATH}"
  mkdir -p "${WASMEDGE_LIB_PATH}"
  mkdir -p "${WASMEDGE_PLUGIN_PATH}"

  mv WasmEdge-${WASMEDGE_VERSION}-Linux/bin/* "${WASMEDGE_BIN_PATH}"/
  if [[ -d "WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge/" ]]; then
    mv WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge/* "${WASMEDGE_PLUGIN_PATH}"/
    rmdir WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge # avoid mv fail when /lib/wasmedge exists
  fi
  mv WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/* "${WASMEDGE_LIB_PATH}"/
  mv libwasmedgePluginWasiNN.so "${WASMEDGE_PLUGIN_PATH}"/

  rm -r WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz WasmEdge-${WASMEDGE_VERSION}-Linux/ WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
}

wasmedge_tflite_deps_init() {
  curl -sLO https://github.com/second-state/WasmEdge-tensorflow-deps/releases/download/${WASMEDGE_VERSION}/WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  tar -zxf WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  mv libtensorflowlite_c.so "${WASMEDGE_LIB_PATH}"/

  rm WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
}

wasmedge_lib_env_init() {
  echo "${WASMEDGE_LIB_PATH}" >/etc/ld.so.conf.d/wasmedge.conf
  ldconfig
}

wasmedge_with_nn_init
wasmedge_tflite_deps_init
# wasmedge_lib_env_init
