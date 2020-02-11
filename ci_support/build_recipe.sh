#!/usr/bin/env bash

set -xe

MYDIR="$(cd "$(dirname "$0")"; pwd -P)"

if [ -n "${TRAVIS}" ];then
    source ci_support/setup_conda.sh
fi

source "${MYDIR}/circleci.sh"

python -m compileall -q .

source "${MYDIR}/jenkinsci.sh"

bash "${MYDIR}/conda.sh"
conda render recipes/eman
conda build purge-all

if [ $AGENT_OS_NAME == "win" ];then
    CONDA_BUILD_TEST="--no-test"
else
    CONDA_BUILD_TEST=""
fi

conda build recipes/eman -c cryoem -c defaults -c conda-forge ${CONDA_BUILD_TEST}
