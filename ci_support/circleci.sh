if [ -n "${CIRCLECI}" ];then
    . $HOME/miniconda/etc/profile.d/conda.sh
    conda activate eman
fi
