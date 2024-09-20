#!/bin/bash

DATA=data
TRAINER=MoCoOp

DATASET=$1
SEED=$2

CFG=$3
SHOTS=16
LOADEP=$4
SUB=all
# SUB=new
EXP=$5


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}_${EXP}
MODEL_DIR=output/base2new/train_base_all/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
--exp ${EXP} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB} \
# DATASET.INCLUDE_ALL_CLASSES True
