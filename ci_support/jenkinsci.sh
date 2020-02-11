if [ -n "$JENKINS_HOME" ];then
    export CPU_COUNT=4
else
    export CPU_COUNT=2
fi
