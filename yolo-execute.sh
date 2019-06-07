#!/bin/bash

# execucao apaga o conteudo dos diretorios em questao! (tudo em 'results')

# vars
# mudar aqui
python2=asd  # se 'true', usa python2 em vez de python3
args="-f"  # <nada> | -f | -n    por enquanto
custompath=true  # se 'true', usa o path em $path
path="../imgs/Projeto_IAPAR/Base_Jersey/IAPAR2_1"
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

# roda yolo-detector.py
file="yolo-detector.py"
mkdir results/YOLO_predictions &&
doit

# roda cut_nasal_pattern.py
file="cut_nasal_pattern.py"
mkdir results/YOLO_nasalpattern &&
doit
