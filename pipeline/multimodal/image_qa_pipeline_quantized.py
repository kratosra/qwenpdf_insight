# qwenpdf_insight/pipeline/multimodal/image_qa_pipeline.py
from __future__ import annotations

import io
import gc
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
import fitz  # PyMuPDF

from transformers import (
    CLIPModel, CLIPProcessor,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers import Qwen2_5_VLForConditionalGeneration

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Fallback si qwen_vl_utils n'est pas importable
# -------------------------------------------------------------------
try:
    from qwen_vl_utils import process_vision_info  # officiel Qwen
except Exception:
    def process_vision_info(messages):
        imgs, vids = [], []
        for msg in messages:
            for block in msg.get("content", []):
                if block.get("type") == "image":
                    im = block.get("image")
                    if isinstance(im, (str, Path)):
                        im = Image.open(im).convert("RGB")
                    imgs.append(im)
                elif block.get("type") == "video":
                    vids.append(block.get("video"))
        return imgs if imgs else None, vids if vids else None

# =========================
# Utils
# =========================
def _free_cuda():
    # Synchroniser puis libérer le cache VRAM
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    gc.collect()

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _is_valid_for_vlm(img_path: Path, min_side: int = 16, max_abs_aspect: float = 180.0) -> bool:
    """Filtre images trop petites ou au ratio extrême (évite l’erreur aspect>200)."""
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if w <= 0 or h <= 0:
            return False
        if min(w, h) < min_side:
            return False
        ar = max(w / h, h / w)
        if ar >= max_abs_aspect:
            return False
        return True
    except Exception:
        return False

def _get_model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cpu")

def _move_batch_to(batch, device):
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

def _flush_vram():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    gc.collect()

# =========================
# 1) Extraction des images
# =========================
def extract_images_to_dir(
    pdf_path: Path,
    out_dir: Path,
    save_full_pages: bool = False,
    page_dpi: int = 180
) -> List[Path]:
    """Extrait images embarquées + (option) rendus de pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    doc = fitz.open(str(pdf_path))

    # a) images embarquées
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images, start=1):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base.get("image", None)
                if not img_bytes:
                    continue
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_path = out_dir / f"p{page_index+1:03d}_img{img_idx:02d}.png"
                pil.save(out_path, format="PNG")
                saved.append(out_path)
            except Exception:
                continue

    # b) rendu de pages (utile pour graphes vectoriels)
    if save_full_pages:
        zoom = page_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = out_dir / f"p{page_index+1:03d}_full.png"
            pix.save(str(out_path))
            saved.append(out_path)

    doc.close()
    return saved

# ==========================================
# 2) Zero-shot image classifier via LAION CLIP
# ==========================================
class ZSImageClassifierHF:
    """
    Zero-shot image classifier (image-only) avec LAION/CLIP.
    NB: On permet de forcer le CPU si la VRAM est limitée.
    """
    def __init__(self, model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(
        self,
        image_paths: List[Path],
        candidate_labels: List[str],
        hypothesis_template: str = "a {} in a financial report"
    ) -> Dict[str, Dict[str, float]]:
        """Retourne {image_path: {label: prob}} pour chaque image."""
        if not image_paths:
            return {}
        prompts = [hypothesis_template.format(lbl) for lbl in candidate_labels]
        results: Dict[str, Dict[str, float]] = {}

        for p in image_paths:
            try:
                image = Image.open(p).convert("RGB")
            except Exception:
                continue
            inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
            # placer les tenseurs sur le même device que le modèle CLIP:
            if self.device == "cuda":
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            out = self.model(**inputs)
            probs = out.logits_per_image.softmax(dim=-1).squeeze(0).cpu().tolist()
            results[str(p)] = {lbl: float(prob) for lbl, prob in zip(candidate_labels, probs)}

        return results

    @staticmethod
    def select_topk(
        zs_scores: Dict[str, Dict[str, float]],
        positive_labels: List[str],
        k: int = 4,
        min_prob: float = 0.18
    ) -> List[Tuple[str, float, str]]:
        """Top-k images selon la meilleure proba parmi positive_labels."""
        ranked: List[Tuple[str, float, str]] = []
        for path, lab2p in zs_scores.items():
            best_label, best_prob = None, 0.0
            for lab in positive_labels:
                p = lab2p.get(lab, 0.0)
                if p > best_prob:
                    best_prob, best_label = p, lab
            if best_prob >= min_prob:
                ranked.append((path, best_prob, best_label or ""))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]

# ==========================================
# 3) Qwen 2.5-VL — chargement 4-bit robuste (bnb 4bit + fallback offload)
# ==========================================
def _load_qwen_vl_4bit(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Charge Qwen2.5-VL en 4-bit via bitsandbytes.
    - Tentative 1: tout sur GPU (rapide)
    - Fallback: device_map='auto' + offload CPU pour éviter l'OOM
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,   # si souci, passer à torch.float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Tentative GPU-only
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        log.info("[Qwen-VL] load 4-bit: GPU-only OK")
        return model, processor
    except Exception as e:
        log.warning(f"[Qwen-VL] GPU-only failed → fallback offload: {e}")

    # Fallback: offload auto (plus lent mais robuste)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    log.info("[Qwen-VL] load 4-bit: auto-offload OK")
    return model, processor

def _safe_generate_vl(model, inputs, max_new_tokens: int) -> List[torch.Tensor]:
    """
    Génération avec retries si OOM: on réduit les tokens et on désactive use_cache.
    + Aligne les entrées sur le device du modèle par sécurité.
    """
    # ✅ Sécurité: réaligner ici aussi
    try:
        model_device = _get_model_device(model)
        inputs = _move_batch_to(inputs, model_device)
    except Exception:
        pass

    try:
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        log.warning("[Qwen-VL] OOM → retry avec max_new_tokens/2 et use_cache=False")
        _free_cuda()
        try:
            with torch.no_grad():
                return model.generate(**inputs, max_new_tokens=max_new_tokens // 2, use_cache=False)
        except Exception:
            log.error("[Qwen-VL] OOM persistant")
            raise

def qwen_vl_answer_from_images(
    image_paths: List[Path],
    question: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 384
) -> str:
    """
    Génère une réponse basée UNIQUEMENT sur les images et la question.
    - Filtre les images (ratio/dim)
    - Qwen-VL 4-bit + fallback offload
    - Aligne les entrées sur le device du modèle avant generate()
    """
    model, processor = _load_qwen_vl_4bit(model_name)

    # Ouvrir + filtrer les images
    pil_images: List[Image.Image] = []
    kept_paths: List[Path] = []
    for p in image_paths:
        try:
            if not _is_valid_for_vlm(p, min_side=16, max_abs_aspect=180.0):
                log.debug(f"[Qwen-VL] skip invalid image for VLM: {p}")
                continue
            pil = Image.open(p).convert("RGB")
            pil_images.append(pil)
            kept_paths.append(p)
        except Exception:
            continue
    if not pil_images:
        return "Aucune image exploitable n’a été trouvée."

    # Messages (multi-images + question)
    sys_msg = system_prompt or (
        "Tu es un analyste attentif et précis. Appuie-toi uniquement sur ce que tu vois dans les images. "
        "Si une donnée est absente/illisible, indique-le clairement. Évite toute hypothèse."
    )
    content_blocks = [{"type": "image", "image": im} for im in pil_images]
    content_blocks.append({"type": "text", "text": f"Question: {question}\nRéponds en français, de façon concise et factuelle."})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user",   "content": content_blocks},
    ]

    # Chat template + vision processing
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Encodage
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # ✅ Aligner les entrées sur le device des embeddings du modèle
    model_device = _get_model_device(model)
    inputs = _move_batch_to(inputs, model_device)

    # Génération (OOM-safe)
    generated_ids = _safe_generate_vl(model, inputs, max_new_tokens=max_new_tokens)

    # Trim + decode
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Nettoyage mémoire
    _free_cuda()
    return answer.strip()

# ==========================================
# 4) Orchestration complète
# ==========================================
DEFAULT_IMAGE_LABELS = [
    "chart", "table", "diagram", "logo", "photo", "map", "screenshot", "flowchart",
    "bar chart", "line chart", "pie chart", "stacked bar chart"
]

def infer_positive_labels_from_question(q: str) -> List[str]:
    ql = q.lower()
    pos = set()
    if any(k in ql for k in ["graph", "graphi", "courbe", "chart", "courbes"]):
        pos.update(["chart", "bar chart", "line chart", "pie chart", "stacked bar chart"])
    if any(k in ql for k in ["table", "tableau", "tab"]):
        pos.update(["table"])
    if any(k in ql for k in ["schéma", "schema", "diagram", "flow"]):
        pos.update(["diagram", "flowchart"])
    if any(k in ql for k in ["logo"]):
        pos.update(["logo"])
    if not pos:
        pos.update(["chart", "table", "diagram"])
    return list(pos)

def multimodal_image_qa_pipeline(
    pdf_path: str | Path,
    user_question: str,
    images_root: str | Path = "data/images",
    k_images: int = 4,
    min_prob: float = 0.18,
    clip_model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    qwen_vl_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    extract_full_pages: bool = False,
    page_dpi: int = 180,
    force_clip_device: Optional[str] = None,   # "cpu" pour épargner la VRAM
) -> Dict[str, object]:
    """
    1) Extrait/charge images (cache sous data/images/<hash>/*.png)
    2) Score zero-shot CLIP (CPU possible)
    3) Sélection top-k
    4) Filtre ratio/dim + fallback rendu de pages si nécessaire
    5) Qwen 2.5-VL (bnb 4-bit + fallback offload) pour générer la réponse
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()

    pdf_hash = file_sha256(pdf_path)
    img_dir = images_root / pdf_hash
    img_dir.mkdir(parents=True, exist_ok=True)

    # Extraction si répertoire vide
    existing = sorted(img_dir.glob("*.png"))
    if not existing:
        existing = extract_images_to_dir(
            pdf_path=pdf_path,
            out_dir=img_dir,
            save_full_pages=extract_full_pages,
            page_dpi=page_dpi
        )

    if not existing:
        return {
            "answer": "Aucune image trouvée dans le document.",
            "selected_images": [],
            "zs_scores": {},
            "positive_labels": [],
            "image_dir": str(img_dir),
        }

    # CLIP zero-shot (autorise CPU si demandé)
    clip_dev = force_clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    zs = ZSImageClassifierHF(model_id=clip_model_id, device=clip_dev)
    zs_scores = zs.score(existing, DEFAULT_IMAGE_LABELS, hypothesis_template="a {} in a financial report")

    positive = infer_positive_labels_from_question(user_question)
    topk = ZSImageClassifierHF.select_topk(zs_scores, positive_labels=positive, k=k_images, min_prob=min_prob)

    # fallback si rien > seuil
    if not topk:
        flat = []
        for pth, lab2p in zs_scores.items():
            if not lab2p:
                continue
            best_lab = max(lab2p.items(), key=lambda x: x[1])
            flat.append((pth, best_lab[1], best_lab[0]))
        flat.sort(key=lambda x: x[1], reverse=True)
        topk = flat[:k_images]

    selected_paths = [Path(p) for (p, _, _) in topk]

    # Filtrer les images invalides AVANT Qwen-VL
    selected_paths = [p for p in selected_paths if _is_valid_for_vlm(p, min_side=16, max_abs_aspect=180.0)]

    # Fallback: rendu de pages si nécessaire
    if not selected_paths and not extract_full_pages:
        try:
            _ = extract_images_to_dir(
                pdf_path=pdf_path,
                out_dir=img_dir,
                save_full_pages=True,
                page_dpi=page_dpi
            )
            existing = sorted(img_dir.glob("*.png"))
            zs_scores = zs.score(existing, DEFAULT_IMAGE_LABELS, hypothesis_template="a {} in a financial report")
            positive = infer_positive_labels_from_question(user_question)
            topk = ZSImageClassifierHF.select_topk(zs_scores, positive_labels=positive, k=k_images, min_prob=min_prob)
            if not topk:
                flat = []
                for pth, lab2p in zs_scores.items():
                    if not lab2p:
                        continue
                    best_lab = max(lab2p.items(), key=lambda x: x[1])
                    flat.append((pth, best_lab[1], best_lab[0]))
                flat.sort(key=lambda x: x[1], reverse=True)
                topk = flat[:k_images]
            selected_paths = [Path(p) for (p, _, _) in topk]
            selected_paths = [p for p in selected_paths if _is_valid_for_vlm(p, min_side=16, max_abs_aspect=180.0)]
        except Exception as _e:
            log.warning(f"[multimodal] page render fallback failed: {_e}")

    if not selected_paths:
        return {
            "answer": "Aucune image exploitable n’a été trouvée.",
            "selected_images": [],
            "zs_scores": zs_scores,
            "positive_labels": positive,
            "image_dir": str(img_dir),
        }

    # Libérer la VRAM avant d’appeler Qwen-VL
    _free_cuda()

    # Qwen-VL réponse (4-bit bnb + fallback offload + retry OOM)
    answer = qwen_vl_answer_from_images(
        image_paths=selected_paths,
        question=user_question,
        model_name=qwen_vl_id
    )

    # Nettoyage final
    _free_cuda()

    return {
        "answer": answer,
        "selected_images": [str(p) for p in selected_paths],
        "zs_scores": zs_scores,
        "positive_labels": positive,
        "image_dir": str(img_dir),
    }

# -----------------------------
# Demo / test rapide en local
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pdf = "pipeline/multimodal/plaquetteOSEAS_FR_2024_simple_page.pdf"  # remplace par un vrai chemin
    question = "Décris précisément les graphiques liés au chiffre d’affaires."
    out = multimodal_image_qa_pipeline(
        pdf_path=test_pdf,
        user_question=question,
        extract_full_pages=False,
        k_images=4,
        min_prob=0.20,
        force_clip_device="cpu",  # ← utile si VRAM serrée
    )
    print("Images retenues:", out["selected_images"])
    print("Labels positifs:", out["positive_labels"])
    print("Réponse:\n", out["answer"])
