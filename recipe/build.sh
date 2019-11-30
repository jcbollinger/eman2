#!/bin/bash

set -xe

${CXX}  -c $SRC_DIR/hello-boost.cpp \
-I$PREFIX/include \
-isystem $PREFIX/include/python2.7 \
-isystem $BUILD_PREFIX/x86_64-conda_cos6-linux-gnu/sysroot/usr/include
#${CXXFLAGS} \
