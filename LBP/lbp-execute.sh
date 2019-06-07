#!/bin/bash

THIS=$(basename $0)

# print help
usage() {
    echo "$THIS -p <path>"
    echo "$THIS -x   usa python2"
    echo "$THIS -h   mostra isso"
    exit 0
}

# roda alguma coisa
doit() {
    echo "\n"$fancy"\n"$cmd $file $args"\n"$fancy"\n"
    $cmd $file $args
}


# defaults
cmd="python3"
file="lbp-detector.py"
fancy="****************************************" # decoracao


# processa argumentos
while getopts "hp:x" OPT; do
    case "$OPT" in
    "h") usage;;
    "p") args="$args -p $OPTARG"; echo "using custom path $OPTARG";;
    "x") cmd="python"; echo "using python2";;
    "?") exit 1;;
    esac
done

rm -r results || true # garante execucao mesmo que a pasta nao exista
mkdir results &&

# roda lbp-detector.py

mkdir results/LBP_predictions &&
mkdir results/LBP_nasalpattern &&
doit
