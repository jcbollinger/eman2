#!/usr/bin/env bash

set -xe

if [ ! -z ${TRAVIS} ];then
    source ci_support/setup_conda.sh

    conda install eman-deps=16.0 boost=1.66 -c cryoem -c defaults -c conda-forge --yes --quiet
fi

if [ ! -z ${CIRCLECI} ];then
    . $HOME/miniconda/etc/profile.d/conda.sh
    conda activate eman-deps-16.0
fi

conda info -a
conda list
conda list --explicit

if [ ! -z "$JENKINS_HOME" ];then
    CPU_COUNT=4
else
    CPU_COUNT=2
fi

build_dir="../build_eman"
src_dir=${PWD}

rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake --version
cmake ${src_dir} -DENABLE_WARNINGS=OFF -DCMAKE_VERBOSE_MAKEFILE=ON
make -j${CPU_COUNT} pyGLUtils2 pyMarchingCubes2
