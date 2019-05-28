#!/bin/bash

# execucao apaga o conteudo dos diretorios em questao! (YOLOv3_predictions, IAPAR2_nasal_pattern)

# vars
# mudar aqui
python2=asd  # se 'true', usa python2 em vez de python3
args=""  # <nada> | -f | -n    por enquanto
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

# roda yolov3_opencv.py
file="yolov3_opencv.py"
rm -r YOLOv3_predictions || true # garante execucao mesmo que a pasta nao exista
mkdir YOLOv3_predictions &&
doit

# roda cut_nasal_pattern.py
file="cut_nasal_pattern.py"
rm -r IAPAR2_nasal_pattern || true
mkdir IAPAR2_nasal_pattern &&
doit
