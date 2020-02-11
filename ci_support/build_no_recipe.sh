#!/usr/bin/env bash

set -xe

MYDIR="$(cd "$(dirname "$0")"; pwd -P)"

if [ -n "${TRAVIS}" ];then
    source ci_support/setup_conda.sh

    conda create -n eman eman-deps-dev=19.0 -c cryoem -c defaults -c conda-forge --yes --quiet
    conda activate eman
fi

source "${MYDIR}/circleci.sh"

python -m compileall -q .

# Build and install eman2
rm -vf ${CONDA_PREFIX}/bin/e2*.py

conda info -a
conda list
conda list --explicit

source "${MYDIR}/jenkinsci.sh"

build_dir="../build_eman"
src_dir=${PWD}

rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake --version
cmake ${src_dir} -DENABLE_WARNINGS=OFF -DENABLE_OPTIMIZE_MACHINE=ON
make -j${CPU_COUNT}
make install

# Run tests
cd "${src_dir}"
bash tests/run_tests.sh
