#!/bin/bash
export TRANSFORMERS_CACHE=/home/ext/konle/sentiment-datasets/bin_cls/models

python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Sadness --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"
python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Joy --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"
python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Love --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"
python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Fear --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"
python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Anger --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"
python3 /home/ext/konle/sentiment-datasets/bin_cls/emotrain.py --target Agitation --batchsize 10 --learningrate 1e-5 --epochs 100 --model "deepset/gbert-large"

