#!/bin/bash

# vars
# mudar aqui
python2=asd  # se 'true', usa python2 em vez de python3
custompath=sdf  # se 'true', usa o path em $path
path="../imgs/~rng"
fancy="****************************************" # decoracao

# nao mudar aqui
cmd="python3"
file=""


# funcao que roda alguma coisa
doit () {
    echo "\n"$fancy"\n"$cmd $file $args"\n"$fancy"\n"
    $cmd $file $args
}

if [ "$python2" = true ]; then
    cmd="python"
    echo "using python2"
fi

if [ "$custompath" = true ]; then
    args="$args -p $path"
    echo "using custom path $path"
fi

rm -r results || true # garante execucao mesmo que a pasta nao exista
mkdir results &&

# roda lbp-detector.py
file="lbp-detector.py"
mkdir results/LBP_predictions &&
mkdir results/LBP_nasalpattern &&
doit
