# iapar-segmentation
se você está aqui, não precisa de uma descrição do projeto.  
neste projeto APENAS material de segmentação (extração da região de interesse)

por motivos tentei fazer compatível com ambos python2 e python3. se tiver a opção, use python3

## tree -d -L 4
estrutura do diretório imgs é a mesma que o Wyverson upou no drive [(link)](https://drive.google.com/drive/folders/112fcVvMraI6m6dWHwDB9heX22jj8m9Jg)
```
.
├── imgs
│   ├── Imagens-USP
│   ├── Projeto_IAPAR
│   │   ├── Base_Jersey
│   │   │   ├── IAPAR2_1
│   │   │   ├── IAPAR2_2
│   │   │   └── IAPAR2_3
│   │   └── Base_Puruna
│   │       ├── Semana1
│   │       ├── Semana2
│   │       └── Semana3
│   └── ~rng  (meu diretório com amostra pequena para testes locais)
├── LBP
│   ├── cfg
│   └── results
│       ├── LBP_nasalpattern
│       └── LBP_predictions
└── YOLO
    ├── cfg
    └── results
        ├── YOLO_nasalpattern
        └── YOLO_predictions
```

## uso
eu só uso os `execute.sh`, mas deve ser tranquilo usar os códigos. códigos aceitam argumentos de linha de comando, ver os `.sh` para exemplos
