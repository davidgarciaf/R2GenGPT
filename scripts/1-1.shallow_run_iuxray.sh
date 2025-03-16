#!/bin/bash
# Entrena el modelo en IU X-ray
# Congela el modelo visual
# No usa LoRA

# Dataset de IU X-ray
dataset="iu_xray"
# Archivos con las anotaciones de reportes médicos
annotation="data/iu_xray/annotation.json"
# Carpeta donde están las imágenes
base_dir="./data/iu_xray/images"
# Nombre de la versión del experimento
version="v1_shallow"
# Carpeta donde se guardan los resultados 
savepath="./save/$dataset/$version"

# Ejecuta el entrenamiento
python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 12 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 2 \
    --max_epochs 15 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    2>&1 |tee -a ${savepath}/log.txt
