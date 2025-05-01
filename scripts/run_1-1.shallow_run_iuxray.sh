#!/bin/bash
# Activar entorno Conda
source /export/anaconda3/bin/activate dgarcia_tfm

# Ir al directorio del proyecto
cd /mnt/sd5/users/dgarcia/R2GenGPT

# Cargar el token de Hugging Face del archivo privado
TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token.env"

# Cargar el token de Hugging Face del archivo privado
if [ -f "$TOKEN_FILE" ]; then
    export $(cat $TOKEN_FILE | xargs)
    echo "Token exportado desde el archivo $TOKEN_FILE"
else
    echo "ERROR: No se encontró el archivo $TOKEN_FILE con el token de Hugging Face."
    exit 1
fi

dataset="iu_xray"
annotation="/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json"
base_dir="/mnt/sd5/users/dgarcia/data/iu_xray/images"

version="v1_shallow"
savepath="/mnt/sd5/users/dgarcia/R2GenGPT/save/$dataset/$version"

python -u /mnt/sd5/users/dgarcia/R2GenGPT/train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 4 \
    --val_batch_size 4 \
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
