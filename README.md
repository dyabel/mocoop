# mocoop
## Installation
pip install -r requirements.txt

Install [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)

## Datasets
Download datasets according to https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md

## RUN
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/base2new_train_moe.sh ucf101 2 vit_b16_c4_ep50_batch32_cls_t2t_10_wcl_25_g1_b_lr32 $exp
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/base2new_test_moe.sh ucf101 2 vit_b16_c4_ep50_batch32_cls_t2t_10_wcl_25_g1_b_lr32 50 $exp
```
