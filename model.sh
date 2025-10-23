#!/bin/bash

TARGET_DIR="/workspace/stable-diffusion-webui/models/Stable-diffusion/"
OUTPUT_FILE="$TARGET_DIR/model.safetensors"
URL="https://civitai.com/api/download/models/2146693?token=acec63d48485726b61aa758b49a415d4"

echo "En attente de la création du dossier : $TARGET_DIR"

# Boucle jusqu'à ce que le dossier existe
while [ ! -d "$TARGET_DIR" ]; do
    sleep 5
done

echo "Dossier détecté, début du téléchargement..."

# Téléchargement avec curl
curl -L "$URL" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Téléchargement terminé : $OUTPUT_FILE"
else
    echo "Erreur lors du téléchargement."
fi
