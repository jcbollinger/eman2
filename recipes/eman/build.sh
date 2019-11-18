#!/bin/bash

set -xe

build_dir="${SRC_DIR}/../build_eman"

rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake --version
cmake $SRC_DIR -DCMAKE_VERBOSE_MAKEFILE=ON

make -j${CPU_COUNT} pyGLUtils2
