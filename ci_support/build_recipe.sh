#!/usr/bin/env bash

set -xe

if [ ! -z ${CIRCLECI} ];then
    . $HOME/miniconda/etc/profile.d/conda.sh
    conda activate eman-deps-16.0
fi

if [ ! -z "$JENKINS_HOME" ];then
    export CPU_COUNT=4
else
    export CPU_COUNT=2
fi

conda info -a
conda list
conda list --explicit

conda build recipes/eman -c cryoem -c defaults -c conda-forge
