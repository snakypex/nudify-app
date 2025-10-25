import argparse
import base64
import io
import json
import logging
import mimetypes
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NudifyProcessor:
    """Classe principale pour le traitement d'images nudify avec upscaling."""

    def __init__(
            self,
            api_url: str = "http://127.0.0.1:7860",
            nudify_api_url: str = "https://nudify-app.com/api/generation/next",
            bearer_token: str = os.getenv("NUDIFY_TOKEN"),
            model_name: str = "model.safetensors",
            max_side: int = 1536,
            min_quality_size: int = 768,
            upscale_factor: int = 2
    ):
        self.api_url = api_url.rstrip('/')
        self.nudify_api_url = nudify_api_url
        self.bearer_token = bearer_token
        self.model_name = model_name
        self.max_side = max_side
        self.min_quality_size = min_quality_size
        self.upscale_factor = upscale_factor

        # Modèle de segmentation chargé une seule fois
        self.segmentation_model = self._load_segmentation_model()
        self.processor = AutoImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")

        # Headers pour l'API nudify
        self.headers = {"bearer-token": bearer_token}

        # Répertoire de travail
        self.work_dir = Path("output")
        self.work_dir.mkdir(exist_ok=True)

    def _load_segmentation_model(self):
        """Charge le modèle de segmentation une seule fois."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        model.eval()
        model.to(device)
        logger.info(f"Modèle de segmentation chargé sur: {device}")
        return model

    @staticmethod
    def round64(x: int) -> int:
        """Arrondit à l'inférieur au multiple de 64 le plus proche."""
        return (x // 64) * 64

    @staticmethod
    def resize_reasonably(img: Image.Image, max_side: int) -> Image.Image:
        """Redimensionne l'image en gardant le ratio, multiples de 64, min 64px."""
        w, h = img.size
        if max(w, h) <= max_side:
            return img

        scale = max_side / max(w, h)
        new_w = max(64, NudifyProcessor.round64(int(w * scale)))
        new_h = max(64, NudifyProcessor.round64(int(h * scale)))

        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def ensure_same_size(mask: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Redimensionne le masque avec NEAREST pour préserver les bords."""
        if mask.size == target_size:
            return mask
        return mask.resize(target_size, Image.NEAREST)

    @staticmethod
    def to_b64(img: Image.Image, fmt: str = "PNG") -> str:
        """Encode une image PIL en base64."""
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def needs_upscaling(self, img: Image.Image) -> bool:
        """Détermine si l'image nécessite un upscaling."""
        w, h = img.size
        min_dim = min(w, h)
        max_dim = max(w, h)

        # Critères de qualité insuffisante
        if min_dim < self.min_quality_size:
            logger.info(f"Image trop petite ({w}x{h}), dimension minimale: {min_dim}px < {self.min_quality_size}px")
            return True

        # Vérifier le ratio de pixels totaux
        total_pixels = w * h
        min_pixels = self.min_quality_size * self.min_quality_size
        if total_pixels < min_pixels:
            logger.info(f"Résolution totale insuffisante: {total_pixels}px < {min_pixels}px")
            return True

        return False

    def upscale_image_esrgan(self, img_path: Path) -> Path:
        """Upscale l'image via l'API A1111 (ESRGAN ou autre upscaler)."""
        try:
            img = Image.open(img_path).convert("RGB")
            img_b64 = self.to_b64(img, "PNG")

            payload = {
                "resize_mode": 0,
                "upscaling_resize": self.upscale_factor,
                "upscaler_1": "R-ESRGAN 4x+",  # ou "ESRGAN_4x", "Lanczos", etc.
                "image": img_b64
            }

            endpoint = f"{self.api_url}/sdapi/v1/extra-single-image"
            logger.info(f"Upscaling de l'image avec facteur {self.upscale_factor}x...")

            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()

            data = response.json()
            if not data.get("image"):
                raise ValueError("Aucune image upscalée retournée")

            # Sauvegarder l'image upscalée
            upscaled_bytes = base64.b64decode(data["image"])
            upscaled_path = self.work_dir / f"upscaled_{img_path.name}"
            upscaled_path.write_bytes(upscaled_bytes)

            upscaled_img = Image.open(upscaled_path)
            logger.info(f"✅ Image upscalée: {img.size} → {upscaled_img.size}")

            return upscaled_path

        except Exception as e:
            logger.error(f"Erreur lors de l'upscaling: {e}")
            logger.warning("Utilisation de l'image originale")
            return img_path

    def upscale_image_basic(self, img_path: Path) -> Path:
        """Upscale basique avec interpolation (si API indisponible)."""
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            new_w = w * self.upscale_factor
            new_h = h * self.upscale_factor

            # Interpolation bicubique de haute qualité
            upscaled = img.resize((new_w, new_h), Image.BICUBIC)

            # Appliquer un léger sharpening
            upscaled_array = np.array(upscaled)
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]]) / 1.0

            for c in range(3):
                upscaled_array[:, :, c] = cv2.filter2D(upscaled_array[:, :, c], -1, kernel)

            upscaled = Image.fromarray(np.clip(upscaled_array, 0, 255).astype(np.uint8))

            upscaled_path = self.work_dir / f"upscaled_basic_{img_path.name}"
            upscaled.save(upscaled_path)

            logger.info(f"✅ Image upscalée (basique): {img.size} → {upscaled.size}")
            return upscaled_path

        except Exception as e:
            logger.error(f"Erreur lors de l'upscaling basique: {e}")
            return img_path

    def preprocess_image(self, img_path: Path) -> Path:
        """Prétraite l'image: upscaling si nécessaire."""
        img = Image.open(img_path).convert("RGB")

        if self.needs_upscaling(img):
            logger.info("🔍 Image de faible qualité détectée, upscaling...")

            # Essayer l'upscaling via API d'abord
            try:
                return self.upscale_image_esrgan(img_path)
            except Exception as e:
                logger.warning(f"Upscaling API échoué, utilisation méthode basique: {e}")
                return self.upscale_image_basic(img_path)
        else:
            logger.info(f"✓ Qualité d'image suffisante ({img.size})")
            return img_path

    def set_model(self):
        url = f"{self.api_url}/sdapi/v1/options"
        payload = {"sd_model_checkpoint": self.model_name}

        while True:
            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                logger.info(f"✅ Modèle chargé: {self.model_name}")
                break
            except requests.RequestException as e:
                logger.warning(f"⚠️ Erreur chargement modèle: {e}")
                time.sleep(2)

    @staticmethod
    def download_image(
            url: str,
            dest_dir: Path,
            filename: Optional[str] = None,
            timeout: int = 30
    ) -> Path:
        """Télécharge une image depuis une URL."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"URL ne renvoie pas une image: {content_type}")

        # Extraction du nom de fichier
        if filename:
            name = filename
        else:
            # Content-Disposition
            cd = response.headers.get("content-disposition", "")
            if "filename=" in cd:
                name = cd.split("filename=")[-1].strip(' "')
            else:
                name = Path(urlparse(url).path).name or "image.jpg"

        # Extension automatique
        if not Path(name).suffix:
            ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
            name += ext

        out_path = dest_dir / name

        # Écriture en streaming
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return out_path

    def generate_clothing_mask(self, image_path: Path, seg_max_size: int = 512) -> np.ndarray:
        """Génère le masque des vêtements."""
        pil_img = Image.open(image_path).convert("RGB")
        original_size = pil_img.size

        # Réduire temporairement pour l'inférence
        if max(original_size) > seg_max_size:
            scale = seg_max_size / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            pil_img_small = pil_img.resize(new_size, Image.LANCZOS)
        else:
            pil_img_small = pil_img

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Inférence sur image réduite
        device = next(self.segmentation_model.parameters()).device
        inputs = self.processor(images=pil_img_small, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.segmentation_model(**inputs)

        # Upsampling directement à la taille originale
        logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # height, width
            mode="bilinear",
            align_corners=False
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()

        # Classes vêtements (hauts, robe, manteau, pantalon, etc.)
        clothing_ids = {4, 5, 6, 7}
        mask = np.isin(pred, list(clothing_ids)).astype(np.uint8) * 255

        # Au lieu de iterations=30 avec kernel 3x3
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        # Un seul passage avec un gros noyau
        kernel_large = np.ones((31, 31), np.uint8)  # Équivalent approximatif
        mask = cv2.dilate(mask, kernel_large, iterations=1)

        clothes_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        # Sauvegarde intermédiaire
        cv2.imwrite(str(self.work_dir / "mask.png"), mask)
        cv2.imwrite(str(self.work_dir / "image_clothes_only.png"), clothes_only)

        return mask

    def generate_inpainted_image(
            self,
            image_path: Path,
            mask_path: Path,
            prompt: str = "nude, breast, pussy",
            negative_prompt: str = "clothes, pant, bad quality"
    ) -> Path:
        """Génère l'image inpaintée via A1111."""

        # Chargement images
        img = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Redimensionnement
        img_resized = self.resize_reasonably(img, self.max_side)
        mask_resized = self.ensure_same_size(mask_img, img_resized.size)

        # Arrondi multiples de 64
        w, h = img_resized.size
        w = max(64, self.round64(w))
        h = max(64, self.round64(h))

        if (w, h) != img_resized.size:
            img_resized = img_resized.resize((w, h), Image.LANCZOS)
            mask_resized = mask_resized.resize((w, h), Image.NEAREST)

        # Encodage base64
        img_b64 = self.to_b64(img_resized)
        mask_b64 = self.to_b64(mask_resized)

        # Payload A1111
        payload = {
            "init_images": [img_b64],
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resize_mode": 0,
            "mask_blur": 4,
            "inpainting_mask_invert": 0,
            "inpainting_fill": 1,
            "inpaint_full_res": False,
            "inpaint_full_res_padding": 32,
            "sampler_name": "DPM++ 3M SDE",
            "scheduler": "Karras",
            "steps": 30,
            "width": w,
            "height": h,
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": -1,  # Aléatoire
            "n_iter": 1,
            "batch_size": 1
        }

        # Appel API
        endpoint = f"{self.api_url}/sdapi/v1/img2img"
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        data = response.json()
        if not data.get("images"):
            raise ValueError("Aucune image retournée par l'API")

        # Sauvegarde
        out_bytes = base64.b64decode(data["images"][0])
        timestamp = int(time.time())
        out_path = self.work_dir / f"result_{timestamp}.png"
        out_path.write_bytes(out_bytes)

        logger.info(f"Image générée: {out_path} ({w}x{h})")
        return out_path

    def process_single_image(self, image_url: str) -> Optional[Path]:
        """Traite une seule image de A à Z avec upscaling si nécessaire."""
        try:
            # 1. Téléchargement
            img_path = self.download_image(image_url, self.work_dir)
            logger.info(f"Image téléchargée: {img_path}")

            # 2. Prétraitement (upscaling si nécessaire)
            processed_path = self.preprocess_image(img_path)

            # 3. Génération masque
            mask_array = self.generate_clothing_mask(processed_path)
            mask_path = self.work_dir / "mask.png"
            cv2.imwrite(str(mask_path), mask_array)

            # 4. Inpainting
            result_path = self.generate_inpainted_image(processed_path, mask_path)

            return result_path

        except Exception as e:
            logger.error(f"Erreur traitement image {image_url}: {e}")
            return None

    def run_continuous(self, delay: float = 1.0):
        """Boucle infinie de traitement."""
        # self.set_model()

        while True:
            try:
                logger.info("Récupération nouvelle image...")

                response = requests.get(self.nudify_api_url, headers=self.headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                image_url = data.get("image_url")
                generation_id = data.get("id")

                if not image_url:
                    logger.warning("Aucune URL d'image trouvée")
                    time.sleep(delay)
                    continue

                logger.info(f"Traitement: {image_url}")
                result = self.process_single_image(image_url)

                if result:
                    logger.info(f"✓ Succès: {result}")
                    image = open(result, "rb")
                    files = {"image": ("mon_image.jpg", image, "image/png")}
                    response = requests.post("https://nudify-app.com/api/generation/done",
                                             data={'generation_id': generation_id}, files=files)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") is True:
                            logger.info(f"Upload done")
                        else:
                            logger.error(f"Erreur traitement image {data}")
                    else:
                        logger.error(f"Une errreur est survenu lors de l'upload: {response.json()}")
                else:
                    logger.error("✗ Échec traitement")

            except requests.RequestException as e:
                logger.error(f"Erreur API nudify: {e}")
                time.sleep(delay)
            except KeyboardInterrupt:
                logger.info("Arrêt demandé par l'utilisateur")
                break
            except Exception as e:
                logger.error(f"Erreur inattendue: {e}")
                time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Nudify Image Processor with Upscaling")
    parser.add_argument("--api-url", default="http://127.0.0.1:17860", help="URL A1111 API")
    parser.add_argument("--model", default="pornmaster_proSDXLV7-inpainting.safetensors", help="Nom du modèle")
    parser.add_argument("--max-side", type=int, default=1536, help="Côté maximum")
    parser.add_argument("--min-quality", type=int, default=768, help="Taille minimale pour bonne qualité")
    parser.add_argument("--upscale-factor", type=int, default=2, choices=[2, 4], help="Facteur d'upscaling")
    parser.add_argument("--delay", type=float, default=1.0, help="Délai entre requêtes (s)")
    parser.add_argument("--single-url", help="URL unique à traiter")

    args = parser.parse_args()

    processor = NudifyProcessor(
        api_url=args.api_url,
        model_name=args.model,
        max_side=args.max_side,
        min_quality_size=args.min_quality,
        upscale_factor=args.upscale_factor
    )

    if args.single_url:
        # Mode unique
        result = processor.process_single_image(args.single_url)
        if result:
            print(f"Résultat: {result}")
            return 0
        else:
            return 1
    else:
        # Mode continu
        processor.run_continuous(delay=args.delay)
        return 0


if __name__ == "__main__":
    sys.exit(main())

