# qwenpdf_insight/pipeline/multimodal/image_qa_pipeline.py
from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
import fitz  # PyMuPDF

from transformers import (
    CLIPModel, CLIPProcessor,
    AutoProcessor,
)
from transformers import Qwen2_5_VLForConditionalGeneration

# -------------------------------------------------------------------
# (Optionnel) Fallback si qwen_vl_utils n'est pas importable
# -------------------------------------------------------------------
try:
    from qwen_vl_utils import process_vision_info  # officiel Qwen
except Exception:
    def process_vision_info(messages):
        """
        Fallback minimal : agrège les images/vidéos à partir de messages.
        Retourne (image_inputs, video_inputs) attendus par le processor Qwen-VL.
        """
        imgs, vids = [], []
        for msg in messages:
            for block in msg.get("content", []):
                if block.get("type") == "image":
                    # block["image"] peut être un PIL.Image ou un chemin
                    im = block.get("image")
                    if isinstance(im, (str, Path)):
                        im = Image.open(im).convert("RGB")
                    imgs.append(im)
                elif block.get("type") == "video":
                    vids.append(block.get("video"))  # laissé tel quel
        return imgs if imgs else None, vids if vids else None


# =========================
# Utils
# =========================
def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# 1) Extraction des images
# =========================
def extract_images_to_dir(
    pdf_path: Path,
    out_dir: Path,
    save_full_pages: bool = False,
    page_dpi: int = 180
) -> List[Path]:
    """
    Extrait toutes les images embarquées du PDF + (option) rendus de pages.
    Sauvegarde en PNG sous out_dir et retourne la liste des chemins.
    """
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
    Zero-shot image classifier (image-only) with LAION/CLIP:
    - laion/CLIP-ViT-B-32-laion2B-s34B-b79K
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
        """
        Retourne {image_path: {label: prob}} pour chaque image.
        """
        if not image_paths:
            return {}
        prompts = [hypothesis_template.format(lbl) for lbl in candidate_labels]
        results: Dict[str, Dict[str, float]] = {}

        for p in image_paths:
            try:
                image = Image.open(p).convert("RGB")
            except Exception:
                continue
            inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.model.device)
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
        """
        Sélectionne top-k images selon la meilleure proba parmi positive_labels.
        Retourne: [(path, best_prob, best_label), ...] trié décroissant.
        """
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
# 3) Qwen 2.5-VL — appel fidèle à l’implémentation
# ==========================================
def qwen_vl_answer_from_images(
    image_paths: List[Path],
    question: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1028
) -> str:
    """
    Génère une réponse basée UNIQUEMENT sur les images et la question.
    Appel strict avec:
      - Qwen2_5_VLForConditionalGeneration
      - AutoProcessor
      - apply_chat_template + process_vision_info
      - trim des tokens d'entrée
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load model / processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else "auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # 2) ouvrir les images
    pil_images: List[Image.Image] = []
    for p in image_paths:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not pil_images:
        return "Aucune image exploitable n’a été trouvée."

    # 3) messages (multi-images + question)
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

    # 4) chat template + vision processing
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # 5) encodage
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if device == "cuda" and torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # 6) génération
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # 7) trim des tokens d'entrée + décodage
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

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
    page_dpi: int = 180
) -> Dict[str, object]:
    """
    1) Extrait/charge images (cache par hash sous data/images/<hash>/*.png)
    2) Score zero-shot CLIP sur labels génériques
    3) Filtre vs question (positive labels) & sélection top-k
    4) Appelle Qwen 2.5-VL (implémentation fidèle) pour générer la réponse
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

    # CLIP zero-shot
    zs = ZSImageClassifierHF(model_id=clip_model_id)
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

    # Qwen-VL réponse (implémentation fidèle)
    answer = qwen_vl_answer_from_images(
        image_paths=selected_paths,
        question=user_question,
        model_name=qwen_vl_id
    )

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
    # Exemple:
    # python -m pipeline.multimodal.image_qa_pipeline
    test_pdf = "pipeline/multimodal/plaquetteOSEAS_FR_2024_simple_page.pdf"  # remplace par un vrai chemin
    question = "Donnez moi une description de l'image bien détaillé"
    out = multimodal_image_qa_pipeline(
        pdf_path=test_pdf,
        user_question=question,
        extract_full_pages=False,   # True si bcp de vecteurs
        k_images=4,
        min_prob=0.20
    )
    print("Images retenues:", out["selected_images"])
    print("Labels positifs:", out["positive_labels"])
    print("Réponse:\n", out["answer"])
