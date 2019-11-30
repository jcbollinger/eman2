#!/bin/bash

set -xe

$BUILD_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++  \
-isystem $PREFIX/include/python2.7 \
-isystem $BUILD_PREFIX/x86_64-conda_cos6-linux-gnu/sysroot/usr/include  \
-I$PREFIX/include \
-c $SRC_DIR/hello-boost.cpp
