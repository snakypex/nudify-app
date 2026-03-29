#!/bin/bash

# =============================================
# SCRIPT DE PROVISIONING COMFYUI
# =============================================
# Ce script ne s'exécute qu'une seule fois.
# Pour forcer une ré-exécution, supprimer le fichier:
#   rm /workspace/.provisioning_done
# =============================================

set -e  # Arrêter en cas d'erreur

WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
NODES_DIR="${COMFYUI_DIR}/custom_nodes"
PROVISIONING_MARKER="${WORKSPACE_DIR}/.provisioning_done"
VENV_PATH="/venv/main/bin/activate"

# --- VÉRIFICATION EXÉCUTION PRÉCÉDENTE ---
if [[ -f "$PROVISIONING_MARKER" ]]; then
    printf "✅ Provisioning déjà effectué le %s\n" "$(cat "$PROVISIONING_MARKER")"
    printf "   Pour forcer une ré-exécution: rm %s\n" "$PROVISIONING_MARKER"
    
    # Activer le venv et démarrer le script Python si nécessaire
    if [[ -f "$VENV_PATH" ]]; then
        source "$VENV_PATH"
    fi
    
    # Vérifier si le script Python tourne déjà
    if ! pgrep -f "python script.py" > /dev/null 2>&1; then
        cd "$WORKSPACE_DIR"
        if [[ -f "script.py" ]]; then
            nohup python script.py > "${WORKSPACE_DIR}/python.log" 2>&1 &
            disown
        fi
    fi
    
    exit 0
fi

# --- ACTIVATION VENV ---
if [[ -f "$VENV_PATH" ]]; then
    source "$VENV_PATH"
else
    printf "⚠️ Environnement virtuel non trouvé: %s\n" "$VENV_PATH"
    printf "   Continuons sans venv...\n"
fi

# --- CONFIGURATION ---

APT_PACKAGES=(
)

PIP_PACKAGES=(
)

NODES=(
)

WORKFLOWS=(
)

CHECKPOINT_MODELS=(
)

UNET_MODELS=(
)

LORA_MODELS=(
    "https://huggingface.co/snakypex/Flux-9B-All-SaaS/resolve/main/Flux%20Klein%20-%20NSFW%20v2.safetensors"
    "https://huggingface.co/snakypex/Flux-9B-All-SaaS/resolve/main/nipplediffusion-f2-klein-9b_v3.safetensors"
)

VAE_MODELS=(
    "https://huggingface.co/snakypex/Flux-9B-All-SaaS/resolve/main/flux2-vae.safetensors"
)

ESRGAN_MODELS=(
)

CONTROLNET_MODELS=(
)

DIFFUSION_MODELS=(
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors"
)

CLIP_MODELS=(
    "https://huggingface.co/snakypex/Flux-9B-All-SaaS/resolve/main/qwen_3_8b_fp8mixed.safetensors"
)

# URL du node Snakypex (définie ici pour éviter erreur variable non définie)
SNK_NODE_URL="${SNK_NODE_URL:-}"

### FONCTIONS ###

function provisioning_start() {
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_pip_packages
    provisioning_get_nodes
    provisioning_get_snk_node
    
    # --- TÉLÉCHARGEMENT DES MODÈLES ---
    local DOWNLOAD_MODE="${DOWNLOAD_MODE:-parallel}"
    
    if [[ "$DOWNLOAD_MODE" == "parallel" ]]; then
        printf "\n🚀 Mode téléchargement PARALLÈLE activé (3 simultanés)\n"
        provisioning_download_parallel "${MODELS_DIR}/loras" "${LORA_MODELS[@]}"
        provisioning_download_parallel "${MODELS_DIR}/controlnet" "${CONTROLNET_MODELS[@]}"
        provisioning_download_parallel "${MODELS_DIR}/vae" "${VAE_MODELS[@]}"
        provisioning_download_parallel "${MODELS_DIR}/upscale_models" "${ESRGAN_MODELS[@]}"
        provisioning_download_parallel "${MODELS_DIR}/diffusion_models" "${DIFFUSION_MODELS[@]}"
        provisioning_download_parallel "${MODELS_DIR}/text_encoders" "${CLIP_MODELS[@]}"
    else
        printf "\n📥 Mode téléchargement SÉQUENTIEL\n"
        provisioning_get_files "${MODELS_DIR}/loras" "${LORA_MODELS[@]}"
        provisioning_get_files "${MODELS_DIR}/controlnet" "${CONTROLNET_MODELS[@]}"
        provisioning_get_files "${MODELS_DIR}/vae" "${VAE_MODELS[@]}"
        provisioning_get_files "${MODELS_DIR}/upscale_models" "${ESRGAN_MODELS[@]}"
        provisioning_get_files "${MODELS_DIR}/diffusion_models" "${DIFFUSION_MODELS[@]}"
        provisioning_get_files "${MODELS_DIR}/text_encoders" "${CLIP_MODELS[@]}"
    fi
    
    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
        printf "📦 Installation des paquets APT...\n"
        sudo apt-get update && sudo apt-get install -y "${APT_PACKAGES[@]}"
    fi
}

function provisioning_get_pip_packages() {
    if [[ ${#PIP_PACKAGES[@]} -gt 0 ]]; then
        printf "📦 Installation/Mise à jour des paquets pip: %s\n" "${PIP_PACKAGES[*]}"
        pip install --no-cache-dir "${PIP_PACKAGES[@]}"
    fi
    
    # Sage Attention - nécessite CUDA toolkit pour compiler
    printf "📦 Installation de SageAttention...\n"
    
    # Méthode 1: Essayer le package pré-compilé sageattention
    if pip install sageattention 2>/dev/null; then
        printf "✅ SageAttention installé via pip\n"
    else
        # Méthode 2: Compiler depuis source (nécessite CUDA toolkit)
        printf "⚠️ Package pré-compilé non disponible, tentative de compilation...\n"
        if command -v nvcc &> /dev/null; then
            if pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation 2>/dev/null; then
                printf "✅ SageAttention compilé depuis source\n"
            else
                printf "⚠️ Échec installation SageAttention - continuera sans (optionnel)\n"
            fi
        else
            printf "⚠️ CUDA toolkit (nvcc) non trouvé - SageAttention ignoré (optionnel)\n"
        fi
    fi
}

function provisioning_get_nodes() {
    if [[ ${#NODES[@]} -eq 0 ]]; then
        printf "\n--- Aucun node à installer ---\n"
        return 0
    fi
    
    printf "\n--- INSTALLATION DES NODES ---\n"
    mkdir -p "$NODES_DIR"
    
    for repo in "${NODES[@]}"; do
        local dir="${repo##*/}"
        # Enlever .git si présent
        dir="${dir%.git}"
        local path="${NODES_DIR}/${dir}"
        local requirements="${path}/requirements.txt"
        
        if [[ -d "$path" ]]; then
            if [[ "${AUTO_UPDATE,,}" != "false" ]]; then
                printf "🔄 Mise à jour du node: %s...\n" "${repo}"
                ( cd "$path" && git pull ) || printf "⚠️ Échec mise à jour: %s\n" "$dir"
                if [[ -f "$requirements" ]]; then
                    printf "🔎 Requirements trouvés pour %s\n" "${dir}"
                    pip install --no-cache-dir -r "$requirements" || printf "⚠️ Échec installation requirements: %s\n" "$dir"
                fi
            fi
        else
            printf "📥 Téléchargement du node: %s...\n" "${repo}"
            if git clone "${repo}" "${path}" --recursive; then
                if [[ -f "$requirements" ]]; then
                    printf "🔎 Requirements trouvés pour %s\n" "${dir}"
                    pip install --no-cache-dir -r "${requirements}" || printf "⚠️ Échec installation requirements: %s\n" "$dir"
                fi
            else
                printf "⚠️ Échec clonage: %s\n" "$repo"
            fi
        fi
    done
}

function provisioning_get_snk_node() {
    # Vérifier si l'URL est définie
    if [[ -z "$SNK_NODE_URL" ]]; then
        printf "ℹ️ SNK_NODE_URL non définie, node Snakypex ignoré\n"
        return 0
    fi
    
    mkdir -p "$NODES_DIR"
    local snk_path="${NODES_DIR}/comfy_ui_res_node.py"
    
    if [[ ! -f "$snk_path" ]]; then
        printf "📥 Téléchargement du node Snakypex (fichier Python)...\n"
        if curl -fsSL -o "$snk_path" "$SNK_NODE_URL"; then
            printf "✨ Node Snakypex téléchargé\n"
        else
            printf "⚠️ Échec téléchargement node Snakypex\n"
        fi
    else
        printf "✅ Node Snakypex déjà présent\n"
    fi
}

function provisioning_get_files() {
    local dir="$1"
    shift
    local arr=("$@")
    
    if [[ ${#arr[@]} -eq 0 ]]; then
        return 0
    fi
    
    mkdir -p "$dir"
    printf "\n📥 Téléchargement de %s modèle(s) vers %s...\n" "${#arr[@]}" "$dir"
    
    for url in "${arr[@]}"; do
        # Décoder le nom de fichier (URL encoded)
        local filename
        filename=$(basename "$url" | sed 's/%20/ /g' | sed 's/%2B/+/g')
        local filepath="${dir}/${filename}"
        
        # Vérifier si le fichier existe déjà
        if [[ -f "$filepath" ]]; then
            printf "✅ Déjà présent: %s\n" "${filename}"
            continue
        fi
        
        printf "📥 Téléchargement: %s\n" "${filename}"
        if provisioning_download "${url}" "${dir}"; then
            printf "✨ Terminé: %s\n" "${filename}"
        else
            printf "⚠️ Échec: %s\n" "${filename}"
        fi
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#          Provisioning container            #\n"
    printf "#                                            #\n"
    printf "#         This will take some time           #\n"
    printf "#                                            #\n"
    printf "# Your container will be ready on completion #\n"
    printf "#                                            #\n"
    printf "##############################################\n\n"
}

function provisioning_print_end() {
    printf "\n✨ Provisioning terminé: L'application va démarrer maintenant\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    
    local url="https://huggingface.co/api/whoami-v2"
    local response
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    [[ "$response" -eq 200 ]]
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    
    local url="https://civitai.com/api/v1/models?hidden=1&limit=1"
    local response
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    [[ "$response" -eq 200 ]]
}

# Download from $1 URL to $2 directory
function provisioning_download() {
    local url="$1"
    local dir="$2"
    local auth_header=""
    
    if [[ -n "$HF_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_header="--header=Authorization: Bearer $HF_TOKEN"
    elif [[ -n "$CIVITAI_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_header="--header=Authorization: Bearer $CIVITAI_TOKEN"
    fi
    
    if [[ -n "$auth_header" ]]; then
        wget "$auth_header" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$dir" "$url"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$dir" "$url"
    fi
}

# Fonction pour téléchargement parallèle (utilise xargs avec 3 workers)
function provisioning_download_parallel() {
    local dir="$1"
    shift
    local urls=("$@")
    
    if [[ ${#urls[@]} -eq 0 ]]; then
        return 0
    fi
    
    mkdir -p "$dir"
    printf "📥 Téléchargement parallèle de %s fichier(s) vers %s...\n" "${#urls[@]}" "$dir"
    
    # Exporter les tokens et variables pour les sous-processus
    export HF_TOKEN CIVITAI_TOKEN dir
    
    # Utiliser xargs pour paralléliser (3 téléchargements simultanés)
    printf '%s\n' "${urls[@]}" | xargs -P 3 -I {} bash -c '
        url="{}"
        # Décoder le nom de fichier
        filename=$(basename "$url" | sed "s/%20/ /g" | sed "s/%2B/+/g")
        filepath="${dir}/${filename}"
        
        if [[ -f "$filepath" ]]; then
            echo "✅ Déjà présent: ${filename}"
            exit 0
        fi
        
        # Déterminer le token d authentification
        auth_header=""
        if [[ -n "$HF_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
            auth_header="--header=Authorization: Bearer $HF_TOKEN"
        elif [[ -n "$CIVITAI_TOKEN" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
            auth_header="--header=Authorization: Bearer $CIVITAI_TOKEN"
        fi
        
        echo "📥 Téléchargement: ${filename}"
        if [[ -n "$auth_header" ]]; then
            wget "$auth_header" -qnc --content-disposition --show-progress -e dotbytes=4M -P "$dir" "$url" 2>/dev/null
        else
            wget -qnc --content-disposition --show-progress -e dotbytes=4M -P "$dir" "$url" 2>/dev/null
        fi
        
        if [[ $? -eq 0 ]]; then
            echo "✨ Terminé: ${filename}"
        else
            echo "⚠️ Échec: ${filename}"
        fi
    '
}

### MAIN ###

# Créer le workspace si nécessaire
mkdir -p "$WORKSPACE_DIR"

# Permettre à l'utilisateur de désactiver le provisioning
if [[ -f "/.noprovisioning" ]]; then
    printf "⏭️ Provisioning désactivé (fichier /.noprovisioning présent)\n"
    exit 0
fi

# Exécuter le provisioning
provisioning_start

# Télécharger les scripts nécessaires
cd "$WORKSPACE_DIR"

cd ComfyUI
git checkout master
git pull
pip install -r requirements.txt

# Marquer le provisioning comme terminé avec timestamp
date "+%Y-%m-%d %H:%M:%S" > "$PROVISIONING_MARKER"
printf "✅ Marqueur de provisioning créé: %s\n" "$PROVISIONING_MARKER"

# Créer le fichier finish.finish
touch "${WORKSPACE_DIR}/finish.finish"

printf "\n🎉 Provisioning terminé avec succès!\n"
exit 0
