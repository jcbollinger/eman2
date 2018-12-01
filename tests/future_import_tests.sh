#!/usr/bin/env bash

future_imports=( 
                print_function
                division
               )

py_files=$( find . -path ./sphire -prune -o -path ./.git -prune -o -not -empty -name '*.py' )

failed_cases=()
for f in ${py_files[@]};do
    echo $f
    for imp in ${future_imports[@]};do
        if [[ $f != "./sphire" && $f != './.git' ]];then
            if ! grep -q "from __future__ import ${imp}" ${f};then
                failed_cases+=("$f")
                echo "  ${f} is missing \"from __future__ import ${imp}"\"
            fi
        fi
    done
done

echo

if [ ${#failed_cases[@]} -ne 0 ];then
    exit 1
fi
