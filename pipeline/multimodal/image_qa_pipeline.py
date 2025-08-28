# qwenpdf_insight/pipeline/multimodal/image_qa_pipeline.py
from __future__ import annotations

import io
import gc
import hashlib
import logging
import json
import re
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from dataclasses import dataclass, field

import torch
from PIL import Image
import fitz  # PyMuPDF

from transformers import (
    CLIPModel, CLIPProcessor,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers import Qwen2_5_VLForConditionalGeneration


# =========================
# Extraction — config & core
# =========================
@dataclass
class ExtractConfig:
    dpi: int = 216                   # rendering resolution for vector pages
    keywords: Tuple[str, ...] = (
        "Évolution", "Evolution", "Focus", "Structure", "KPIs", "CA",
        "Auto Mono", "Auto global", "Réseau", "Branche", "Segment",
        "Tendance", "Par mois", "Cumulé", "Région", "Usage"
    )
    expand_left: float = 50.0        # px expansion around keyword bbox (at 72dpi)
    expand_top: float = 80.0
    expand_right: float = 500.0
    expand_bottom: float = 350.0
    min_clip_area: float = 10_000.0  # ignore tiny clips
    page_mask: Optional[Iterable[int]] = None  # 1-based page indices to process; None = all
    out_dir_pages: Optional[Path] = None       # where to save full-page renders (png)
    out_dir_figs: Optional[Path] = None        # where to save cropped figures (png)


@dataclass
class ExtractResult:
    page_pngs: List[Path] = field(default_factory=list)
    figure_pngs: List[Path] = field(default_factory=list)
    embedded_pngs: List[Path] = field(default_factory=list)


def _ensure_dirs(cfg: ExtractConfig):
    if cfg.out_dir_pages:
        cfg.out_dir_pages.mkdir(parents=True, exist_ok=True)
    if cfg.out_dir_figs:
        cfg.out_dir_figs.mkdir(parents=True, exist_ok=True)


def _render_page_to_png(page: "fitz.Page", dpi: int) -> bytes:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _save_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def extract_pdf_figures(pdf_path: Path, cfg: Optional[ExtractConfig] = None) -> ExtractResult:
    """
    Hybrid extractor:
    1) export embedded images (rare in vector dashboards),
    2) render pages and crop around keyword hits (vector charts),
    3) optionally save full-page renders.

    Écrit aussi figures/figures_index.json avec (png_path, page, bbox_72dpi, dpi)
    pour permettre l'évaluation VLM ↔ zone PDF.
    """
    if cfg is None:
        cfg = ExtractConfig()
    _ensure_dirs(cfg)

    doc = fitz.open(str(pdf_path))
    result = ExtractResult()
    index_entries = []  # <-- collecte des métadonnées de crops

    # (A) Extract embedded images (bitmaps) if present
    for pidx in range(len(doc)):
        page = doc[pidx]
        for i, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base.get("image")
            if not img_bytes:
                continue
            # force png to standardize output
            out = Path(cfg.out_dir_figs or pdf_path.with_suffix("")) / f"page{pidx+1:02d}_embedded_{i:02d}.png"
            _save_bytes(out, img_bytes)
            result.embedded_pngs.append(out)

    # (B) Render pages (handles vector content) and detect crop regions via keywords
    zoom = cfg.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pages_to_process = cfg.page_mask if cfg.page_mask else range(1, len(doc) + 1)

    for idx in pages_to_process:
        page = doc[idx - 1]

        # Save full page render if requested
        if cfg.out_dir_pages is not None:
            page_png = _render_page_to_png(page, cfg.dpi)
            outp = cfg.out_dir_pages / f"page_{idx:02d}.png"
            _save_bytes(outp, page_png)
            result.page_pngs.append(outp)

        # Find keyword bboxes at 72dpi space then expand
        rects = []
        for kw in cfg.keywords:
            for r in page.search_for(kw, quads=False):
                R = fitz.Rect(r)
                R = fitz.Rect(
                    R.x0 - cfg.expand_left,
                    R.y0 - cfg.expand_top,
                    R.x1 + cfg.expand_right,
                    R.y1 + cfg.expand_bottom,
                )
                R = R & page.rect
                if R.get_area() >= (cfg.min_clip_area / (zoom * zoom)):  # scale area threshold back to 72dpi coords
                    rects.append(R)

        # Merge overlapping rects (simple greedy union)
        merged = []
        for R in rects:
            merged_any = False
            for j, M in enumerate(merged):
                if R.intersects(M):
                    merged[j] = M | R
                    merged_any = True
                    break
            if not merged_any:
                merged.append(R)

        # Render each merged region
        for j, clip in enumerate(merged, start=1):
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            out = (cfg.out_dir_figs or pdf_path.with_suffix("")).joinpath(f"page_{idx:02d}_fig_{j:02d}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            pix.save(out.as_posix())
            result.figure_pngs.append(out)

            # métadonnées pour l'évaluation (bbox en coordonnées 72dpi)
            index_entries.append({
                "png_path": str(out),
                "page_index_1based": idx,
                "bbox_72dpi": [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)],
                "dpi": cfg.dpi,
            })

    # Sauvegarde de l'index des figures
    if cfg.out_dir_figs is not None:
        idx_json = (cfg.out_dir_figs / "figures_index.json")
        try:
            with open(idx_json, "w", encoding="utf-8") as f:
                json.dump(index_entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"[extract] Impossible d'écrire figures_index.json: {e}")

    doc.close()
    return result


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

def _pil_downscale_max_side(p: Path, max_side: int = 1536) -> Optional[Image.Image]:
    """Charge et downscale (long côté <= max_side) pour réduire VRAM/Tensor size."""
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return im
    except Exception:
        return None


# =========================
# Extraction helpers
# =========================
def extract_images_to_dir(
    pdf_path: Path,
    out_dir: Path,
    save_full_pages: bool = False,
    page_dpi: int = 180
) -> List[Path]:
    """Ancienne implémentation (bitmaps embarqués + pages rendues).
       Conservée comme fallback si le module hybride n’est pas dispo.
       Toutes les images sont converties en PNG.
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

    # b) rendu de pages
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


def robust_extract_images(
    pdf_path: Path,
    cache_dir: Path,
    dpi: int = 216,
    want_pages: bool = True,
    keywords: Optional[Tuple[str, ...]] = (
        "Évolution","Evolution","Focus","Structure","KPIs","CA",
        "Auto Mono","Auto global","Réseau","Branche","Segment","Par mois","Cumulé","Région","Usage"
    )
) -> Dict[str, List[Path]]:
    """
    Retourne un dict avec:
      - 'figures': figures recadrées (vectoriel → rendu + crop par mots-clés)
      - 'embedded': images embarquées
      - 'pages': pages rendues (si want_pages=True)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Ici on appelle extract_pdf_figures (défini dans ce fichier)
    if extract_pdf_figures and ExtractConfig:
        cfg = ExtractConfig(
            dpi=dpi,
            out_dir_pages=(cache_dir / "pages") if want_pages else None,
            out_dir_figs=(cache_dir / "figures"),
            keywords=keywords,
        )
        res = extract_pdf_figures(pdf_path, cfg)
        return {
            "figures": sorted(res.figure_pngs),
            "embedded": sorted(res.embedded_pngs),
            "pages": sorted(res.page_pngs) if want_pages else [],
        }

    # Fallback si jamais indispo
    log.warning("[extract] extract_pdf_figures indisponible — fallback simple.")
    paths = extract_images_to_dir(pdf_path, cache_dir, save_full_pages=want_pages, page_dpi=dpi)
    figures = [p for p in paths if p.name.endswith("_img01.png")]  # simple heuristique
    pages = [p for p in paths if p.name.endswith("_full.png")]
    embedded = [p for p in paths if p not in pages]
    return {"figures": sorted(figures), "embedded": sorted(embedded), "pages": sorted(pages)}


# ==========================================
# 2) Zero-shot image classifier via LAION CLIP
# ==========================================
class ZSImageClassifierHF:
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
# 3) Qwen 2.5-VL — chargement 4-bit robuste
# ==========================================
def _load_qwen_vl_4bit(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
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
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=max_new_tokens // 2, use_cache=False)


# ==========================================
# 3bis) VLM séquentiel pour économiser la VRAM
# ==========================================
def qwen_vl_answers_sequential(
    image_paths: List[Path],
    question: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 384,
    max_side_px: int = 1536,   # ↓ dimension max pour réduire VRAM
) -> List[Tuple[str, str]]:
    """
    Retourne une liste [(image_path, answer), ...].
    Charge Qwen-VL une seule fois, et traite les images 1 par 1
    en vidant le cache CUDA entre chaque itération.
    """
    model, processor = _load_qwen_vl_4bit(model_name)

    sys_msg = system_prompt or (
        "Tu es un analyste attentif et précis. Appuie-toi uniquement sur ce que tu vois dans l'image. "
        "Si une donnée est absente/illisible, indique-le clairement. Évite toute hypothèse."
    )

    outputs: List[Tuple[str, str]] = []
    for p in image_paths:
        pil_img = _pil_downscale_max_side(p, max_side=max_side_px)
        if pil_img is None:
            outputs.append((str(p), "Image illisible."))
            continue

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
            {"role": "user",   "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": f"Question: {question}\nRéponds en français, de façon concise et factuelle."}
            ]},
        ]

        try:
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_device = _get_model_device(model)
            inputs = _move_batch_to(inputs, model_device)

            generated_ids = _safe_generate_vl(model, inputs, max_new_tokens=max_new_tokens)
            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            outputs.append((str(p), answer.strip()))
        except Exception as e:
            outputs.append((str(p), f"Erreur génération: {e}"))
        finally:
            # ↓↓↓ Libération mémoire entre images ↓↓↓
            try:
                del inputs, pil_img, image_inputs, video_inputs
            except Exception:
                pass
            _free_cuda()

    # Libération finale (optionnel)
    _free_cuda()
    return outputs


# ==========================================
# 4) Orchestration complète — figures-only + VRAM friendly
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
    extract_full_pages: bool = False,  # conservé pour compat, non utilisé ici
    page_dpi: int = 216,
    force_clip_device: Optional[str] = None,   # "cpu" pour épargner la VRAM
) -> Dict[str, object]:
    """
    Pipeline VRAM-friendly:
      1) Extraction → figures uniquement (crops)
      2) CLIP ZS pour prioriser
      3) VLM (Qwen-VL) image-par-image + vidage CUDA entre chaque itération
      4) Retourne un output détaillé par image
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()

    pdf_hash = file_sha256(pdf_path)
    img_dir = images_root / pdf_hash
    img_dir.mkdir(parents=True, exist_ok=True)

    # === Extraction (figures only)
    extracted = robust_extract_images(
        pdf_path=pdf_path,
        cache_dir=img_dir,
        dpi=page_dpi,
        want_pages=True,  # utile pour debug mais non exploité par le VLM
    )

    # N'utiliser que figures/
    candidate_paths: List[Path] = [Path(p) for p in extracted.get("figures", []) if Path(p).exists()]

    # Fallback "relaxé" si aucune figure trouvée : relance extraction avec paramètres plus permissifs
    if not candidate_paths:
        log.info("[extract] Aucune figure trouvée. Relance extraction avec paramètres relaxés...")
        relaxed_keywords = (
            "Évolution","Evolution","Focus","Structure","KPIs","CA","Courbe","Courbes","Graph","Graphique",
            "Auto","Mono","Flotte","Réseau","Branche","Segment","Par mois","Cumulé","Région","Usage","Tendance",
            "Tableau","Performance","Performants","Sinistrés","Diversification"
        )
        try:
            cfg_relax = ExtractConfig(
                dpi=max(page_dpi, 240),          # ↑ dpi pour des crops plus nets
                keywords=relaxed_keywords,        # + de mots-clés FR/EN
                expand_left=80.0,                 # ↑ expansions autour des mots-clés
                expand_top=120.0,
                expand_right=640.0,
                expand_bottom=420.0,
                min_clip_area=6_000.0,            # ↓ seuil mini de zone
                out_dir_pages=(img_dir / "pages"),
                out_dir_figs=(img_dir / "figures")
            )
            res_relax = extract_pdf_figures(pdf_path, cfg_relax)
            candidate_paths = [Path(p) for p in res_relax.figure_pngs if Path(p).exists()]
        except Exception as e:
            log.warning(f"[extract] Fallback relaxé impossible: {e}")

    if not candidate_paths:
        return {
            "answer": "Aucune figure exploitable n’a été trouvée (dossier figures vide après double passage).",
            "selected_images": [],
            "zs_scores": {},
            "positive_labels": infer_positive_labels_from_question(user_question),
            "image_dir": str(img_dir),
            "per_image": [],
        }

    # =========================
    # CLIP zero-shot (priorisation)
    # =========================
    clip_dev = force_clip_device or ("cuda" if torch.cuda.is_available() else "cpu")
    zs = ZSImageClassifierHF(model_id=clip_model_id, device=clip_dev)
    zs_scores = zs.score(candidate_paths, DEFAULT_IMAGE_LABELS, hypothesis_template="a {} in a financial report")

    positive = infer_positive_labels_from_question(user_question)
    topk = ZSImageClassifierHF.select_topk(zs_scores, positive_labels=positive, k=k_images, min_prob=min_prob)

    # fallback si rien > seuil : prendre les meilleures globales
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

    if not selected_paths:
        return {
            "answer": "Aucune figure exploitable n’a passé les filtres (taille/ratio).",
            "selected_images": [],
            "zs_scores": zs_scores,
            "positive_labels": positive,
            "image_dir": str(img_dir),
            "per_image": [],
        }

    _free_cuda()

    # --------------------------------
    # VLM: traitement 1 image à la fois
    # --------------------------------
    def best_score_for(pth: Path) -> float:
        s = zs_scores.get(str(pth), {})
        return max(s.values()) if s else 0.0
    selected_paths.sort(key=best_score_for, reverse=True)

    per_image_answers = qwen_vl_answers_sequential(
        image_paths=selected_paths,
        question=user_question,
        model_name=qwen_vl_id,
        max_new_tokens=384,
        max_side_px=1536,   # baisser à 1280 si VRAM très limitée
    )

    # Construire un output par image avec label+score CLIP
    per_image_outputs = []
    for p, ans in per_image_answers:
        lab2p = zs_scores.get(p, {})
        best_lab = None
        best_prob = 0.0
        for lab in positive:
            v = lab2p.get(lab, 0.0)
            if v > best_prob:
                best_prob, best_lab = v, lab
        if best_lab is None and lab2p:
            best_lab, best_prob = max(lab2p.items(), key=lambda x: x[1])

        per_image_outputs.append({
            "image": p,
            "best_label": best_lab or "",
            "best_score": float(best_prob),
            "answer": ans,
        })

    _free_cuda()

    return {
        "answer": "Réponses générées image par image (voir 'per_image').",
        "selected_images": [str(p) for p in selected_paths],
        "zs_scores": zs_scores,
        "positive_labels": positive,
        "image_dir": str(img_dir),
        "per_image": per_image_outputs,
    }


# =========================
# Évaluation VLM ↔ PDF (zone de la figure)
# =========================
_num_re = re.compile(r"[-+]?\d+(?:[\.,]\d+)?%?")

def _norm(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w,\.\-\+%/ ]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def _f1_tokens(ref_toks: List[str], hyp_toks: List[str]) -> Tuple[float, float, float]:
    ref_set, hyp_set = set(ref_toks), set(hyp_toks)
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    f1 = 2*p*r / max(1e-9, (p+r))
    return p, r, f1

def _extract_numbers(text: str) -> List[str]:
    nums = _num_re.findall(text)
    return [n.replace(",", ".") for n in nums]

def _num_compare(ref_nums: List[str], hyp_nums: List[str], tol: float = 0.01) -> Dict[str, float]:
    def parse(x: str):
        if x.endswith("%"):
            try:
                return ("pct", float(x[:-1]))
            except:
                return ("raw", x)
        try:
            return ("num", float(x))
        except:
            return ("raw", x)

    ref = [parse(x) for x in ref_nums]
    hyp = [parse(x) for x in hyp_nums]
    matched = 0
    for t, v in hyp:
        ok = False
        for t2, u in ref:
            if t == "raw" or t2 == "raw":
                if str(v) == str(u):
                    ok = True; break
            elif t == t2 == "pct":
                if abs(v - u) <= max(0.1, tol*max(abs(u),1.0)):
                    ok = True; break
            elif t == t2 == "num":
                if abs(v - u) <= tol*max(abs(u),1.0):
                    ok = True; break
        if ok:
            matched += 1
    hyp_total = max(1, len(hyp))
    ref_total = max(1, len(ref))
    return {
        "hyp_in_ref_ratio": matched / hyp_total,
        "ref_covered_ratio": matched / ref_total,
        "hyp_count": len(hyp),
        "ref_count": len(ref),
    }

def _extract_ref_text_from_bbox(pdf_path: Path, page_1based: int, bbox_72dpi: List[float]) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_1based - 1]
    rect = fitz.Rect(*bbox_72dpi)
    text = page.get_text("text", clip=rect)
    doc.close()
    return text or ""


# -----------------------------
# Demo / test rapide en local
# -----------------------------
if __name__ == "__main__":
    test_pdf = "pipeline/multimodal/DocFinancier.pdf"  # remplace par un vrai chemin
    question = """Tu es un extracteur visuel strict. 
Analyse uniquement l’image fournie et décris TOUTES les informations visibles 
sans interprétation ni résumé. 

Règles :
- Liste toutes les valeurs numériques significatives (montants, pourcentages, taux, années). 
- Préserve l’organisation logique (par ligne/colonne si c’est un tableau, par catégorie si c’est un graphe).
- N’ajoute aucune analyse ni hypothèse, contente-toi de rapporter fidèlement ce qui est écrit ou montré.
- Si une valeur est illisible, écris "[illisible]".

Question : Donne une description exhaustive de l'image en citant toutes les valeurs numériques significatives, 
de façon structurée et fidèle."""


    # 0) Lancement du pipeline
    out = multimodal_image_qa_pipeline(
        pdf_path=test_pdf,
        user_question=question,
        k_images=7,
        min_prob=0.20,
        page_dpi=216,
        force_clip_device="cpu",  # CLIP sur CPU pour économiser la VRAM
    )

    # 1) Affichage console
    print("Images retenues:", out["selected_images"])
    print("Labels positifs:", out["positive_labels"])
    print("Réponses par image:")
    for item in out.get("per_image", []):
        print("-" * 40)
        print("Image:", item["image"])
        print("Label/Score CLIP:", item["best_label"], item["best_score"])
        print("Réponse:\n", item["answer"])

    # 2) Sauvegarde lisible dans un fichier texte
    txt_path = Path("data/stats/results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PDF: {test_pdf}\n")
        f.write(f"Question: {question}\n\n")
        f.write("Images retenues:\n")
        for p in out["selected_images"]:
            f.write(f" - {p}\n")
        f.write("\nLabels positifs: " + ", ".join(out["positive_labels"]) + "\n\n")
        f.write("Réponses par image:\n")
        for item in out.get("per_image", []):
            f.write("-" * 60 + "\n")
            f.write(f"Image: {item['image']}\n")
            f.write(f"Label/Score CLIP: {item['best_label']} ({item['best_score']:.3f})\n")
            f.write(f"Réponse:\n{item['answer']}\n\n")

    # 3) Sauvegarde brute en JSON
    json_path = Path("data/stats/results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats enregistrés dans:\n - {txt_path}\n - {json_path}")

    # 4) Évaluation VLM ↔ PDF (zone)
    #    On retrouve le figures_index.json dans le cache du PDF courant:
    images_root = Path(out["image_dir"])
    figures_index = images_root / "figures" / "figures_index.json"
    if figures_index.exists():
        try:
            index = json.loads(figures_index.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[eval] Impossible de lire {figures_index}: {e}")
            index = []
    else:
        print(f"[eval] {figures_index} introuvable — as-tu bien exécuté l'extraction ?")
        index = []

    # Map png -> meta
    meta_by_png = {entry["png_path"]: entry for entry in index}

    rows = []
    for item in out.get("per_image", []):
        png = item["image"]
        ans = item.get("answer", "")
        meta = meta_by_png.get(png)

        if not meta:
            rows.append({
                "image": png,
                "status": "NO_META",
                "f1": 0.0, "rougeL": 0.0, "p": 0.0, "r": 0.0,
                "num_hyp_in_ref": 0.0, "num_ref_covered": 0.0,
                "ref_len": 0, "hyp_len": len(_norm(ans)),
                "ref_text": "", "hyp_text": ans,
            })
            continue

        ref_text = _extract_ref_text_from_bbox(Path(test_pdf), meta["page_index_1based"], meta["bbox_72dpi"])
        ref_toks = _norm(ref_text)
        hyp_toks = _norm(ans)

        p, r, f1 = _f1_tokens(ref_toks, hyp_toks)
        lcs_len = _lcs(ref_toks, hyp_toks)
        rougeL = 2*lcs_len / max(1, len(ref_toks) + len(hyp_toks))

        nums_ref = _extract_numbers(ref_text)
        nums_hyp = _extract_numbers(ans)
        num_stats = _num_compare(nums_ref, nums_hyp, tol=0.01)

        rows.append({
            "image": png,
            "status": "OK",
            "f1": round(f1, 4),
            "rougeL": round(rougeL, 4),
            "p": round(p, 4),
            "r": round(r, 4),
            "num_hyp_in_ref": round(num_stats["hyp_in_ref_ratio"], 4),
            "num_ref_covered": round(num_stats["ref_covered_ratio"], 4),
            "ref_len": len(ref_toks),
            "hyp_len": len(hyp_toks),
            "ref_text": ref_text[:8000],
            "hyp_text": ans[:8000],
        })

    # Sauvegardes des rapports d'évaluation
    if rows:
        Path("data/stats/eval_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        with open("data/stats/eval_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image","status","f1","rougeL","p","r","num_hyp_in_ref","num_ref_covered","ref_len","hyp_len"])
            for r in rows:
                w.writerow([r["image"], r["status"], r["f1"], r["rougeL"], r["p"], r["r"], r["num_hyp_in_ref"], r["num_ref_covered"], r["ref_len"], r["hyp_len"]])
        with open("data/stats/eval_report.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write("="*80 + "\n")
                f.write(f"Image: {r['image']} | Status: {r['status']}\n")
                f.write(f"F1={r['f1']}  ROUGE-L={r['rougeL']}  P={r['p']}  R={r['r']}\n")
                f.write(f"Num match (hyp∈ref)={r['num_hyp_in_ref']}  Couverture ref={r['num_ref_covered']}\n")
                f.write("- Référence (PDF zone):\n")
                f.write(r["ref_text"] + "\n")
                f.write("- Réponse VLM:\n")
                f.write(r["hyp_text"] + "\n\n")
        print("Évaluation OK -> eval_report.json / eval_report.csv / eval_report.txt")
    else:
        print("Aucune donnée d'évaluation à sauvegarder (pas de per_image ou pas d'index).")
