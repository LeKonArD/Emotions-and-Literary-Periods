#!/bin/bash
export TRANSFORMERS_CACHE=/home/ext/konle/diss
python3 /home/ext/konle/diss/code/pretrainer.py -output "/home/ext/konle/diss/logs_drama.tsv" -modelname "deepset/gbert-large" -pretrainlr 2e-5 -pretrainbsize 10 -pretrainsteps 100000 -evalinterval 50 -pretrainmethod "bert" -pretrainfile "poems.txt" -evalbsize 30 -clsbsize 20 -clslr 1e-5 -clsepochs 10
