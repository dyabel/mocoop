#!/bin/bash

DATA=data
TRAINER=MoCoOp

DATASET=$1
SEED=$2

CFG=$3
SHOTS=16
EXP=$4

DIR=output/base2new/train_base_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}_${EXP}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--exp ${EXP} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all

