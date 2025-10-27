#!/bin/bash
set -e

# SRC_ROOT_DATA="$1"
SRC_DATA="$1"
# echo $SRC_ROOT_DATA
# echo $SRC_DATA 
eval "$(conda shell.bash hook)"
# conda init
conda activate TreeLearn
# python3 tools/evaluation/evaluate.py --config configs/evaluation/evaluate.yaml --src_data $SRC_DATA
python3 tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml --src_data $SRC_DATA
# conda init
conda activate pdm_env
# python3 ./inference.py --src_root_data $SRC_ROOT_DATA --src_data $SRC_DATA