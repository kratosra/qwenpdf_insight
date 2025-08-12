# app/qwen_interface_unified.py

import time, shutil, logging, os, json, uuid
from pathlib import Path
import gradio as gr
import torch  # auto-detect device

from pipeline.cache_manager import EmbeddingCacheManager
from pipeline.extract.extract_with_docling import pdf_to_markdown as pdf_to_md_text, chunk_markdown as chunk_text
from pipeline.extract.extract_with_docling_img import pdf_to_markdown as pdf_to_md_img,  chunk_markdown as chunk_img
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_smart, classify_prompt_zero_shot

# ─────────────────────────────────────────────────────────────────────────────
# Logging (terminal only)
# ─────────────────────────────────────────────────────────────────────────────
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOGLEVEL, logging.INFO),
                    format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def _rid() -> str:
    return uuid.uuid4().hex[:8]

def _short(txt: str, n: int = 90) -> str:
    if not isinstance(txt, str):
        return str(txt)
    return txt if len(txt) <= n else txt[:n] + "…"

# ─────────────────────────────────────────────────────────────────────────────
# Constantes (fixes → n’affectent PAS le cache)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 2048
SLIDING_OVERLAP     = 200
# valeurs par défaut des réglages "safe"
DEF_TOP_K           = 3
DEF_MIN_CLIP_PROB   = 0.18
DEF_PAGE_DPI        = 180
DEF_EXTRACT_PAGES   = False
DEF_K_IMAGES        = 4
DEF_CLIP_MODEL_ID   = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEF_QWEN_VL_ID      = "Qwen/Qwen2.5-VL-3B-Instruct"
DEF_QWEN_EMBED      = "Qwen/Qwen3-Embedding-0.6B"
DEF_QWEN_LLM        = "Qwen/Qwen3-0.6B"
DEF_DEVICE          = "auto"  # auto/cuda/cpu

# ─────────────────────────────────────────────────────────────────────────────
# Dossiers de travail
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("data")
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
MD_DIR       = PROJECT_ROOT / "markdown"
CHUNK_DIR    = PROJECT_ROOT / "chunks"
LOG_DIR      = PROJECT_ROOT / "logs"
IMG_DIR_ROOT = PROJECT_ROOT / "images"   # images VLM; le cache VLM est vidé en mode ++
LOG_FILE     = LOG_DIR / "interface_log.txt"
for d in (UPLOAD_DIR, MD_DIR, CHUNK_DIR, LOG_DIR, IMG_DIR_ROOT):
    d.mkdir(parents=True, exist_ok=True)
if not LOG_FILE.exists():
    LOG_FILE.write_text("", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Caches (selon mode)
#   - text-only     -> data/cache
#   - text-image/++ -> data/cache_img
# ─────────────────────────────────────────────────────────────────────────────
cache_text = EmbeddingCacheManager(cache_dir="data/cache")
cache_img  = EmbeddingCacheManager(cache_dir="data/cache_img")

MODES = ["text-only", "text-image-with-docling", "text-image++"]

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_uploaded(pdf_file, mode: str, rid: str) -> Path:
    src = Path(pdf_file.name)
    cm = cache_text if mode == "text-only" else cache_img
    file_hash = cm._pdf_hash(src)
    local_pdf = UPLOAD_DIR / f"{file_hash}.pdf"
    if not local_pdf.exists():
        shutil.copy(src, local_pdf)
        log.info(f"[{rid}] ▶ Upload: {src.name} -> {local_pdf.name} (hash={file_hash[:8]})")
    else:
        log.info(f"[{rid}] ▶ Upload: déjà présent (hash={file_hash[:8]}) -> {local_pdf.name}")
    return local_pdf

def _prepare_chunks(local_pdf: Path, mode: str, rid: str):
    """
    Prépare chunks/embeddings/index selon le mode.
    NB: on NE stocke PAS l'index; on le reconstruit après chargement si besoin.
    """
    if mode == "text-only":
        cache = cache_text
        to_md, chunker = pdf_to_md_text, chunk_text
        log.info(f"[{rid}] ► Mode=TEXT-ONLY | cache_dir={cache.cache_dir}")
    else:
        cache = cache_img
        to_md, chunker = pdf_to_md_img, chunk_img
        log.info(f"[{rid}] ► Mode=TEXT-IMAGE | cache_dir={cache.cache_dir}")

    # cache
    t0 = time.time()
    cached = cache.load_cache(local_pdf)
    if cached:
        chunks, embs, index = cached
        took = time.time() - t0
        log.info(f"[{rid}]   Cache HIT en {took:.3f}s | chunks={len(chunks)} emb_shape={getattr(embs,'shape',None)}")
        if (index is None) and embs is not None and len(embs):
            log.debug(f"[{rid}]   Index absent → reconstruction")
            index = build_faiss_index(embs)
        return (chunks, embs, index), cache, {"used_cache": True, "chunks": len(chunks)}

    # pas de cache → pipeline
    log.info(f"[{rid}]   Cache MISS → extraction Docling …")
    t0 = time.time()
    md_path = to_md(local_pdf, output_dir=MD_DIR)
    log.info(f"[{rid}]     Docling OK en {time.time()-t0:.2f}s | md={md_path.name}")

    chunk_file = CHUNK_DIR / f"{md_path.stem}_chunks.txt"
    t1 = time.time()
    chunks = chunker(md_path, chunk_size=CHUNK_SIZE, output_chunks_path=chunk_file, sliding_overlap=SLIDING_OVERLAP)
    log.info(f"[{rid}]     Chunking: {len(chunks)} chunks en {time.time()-t1:.2f}s")

    if not chunks:
        log.warning(f"[{rid}]   Aucun chunk généré")
        return ([], [], None), cache, {"used_cache": False, "chunks": 0}

    t2 = time.time()
    embs = embed_chunks_qwen3(chunks)
    log.info(f"[{rid}]     Embeddings: shape={getattr(embs,'shape',None)} en {time.time()-t2:.2f}s")

    t3 = time.time()
    index = build_faiss_index(embs)
    log.info(f"[{rid}]     Index: construit en {time.time()-t3:.2f}s")

    cache.save_cache(local_pdf, chunks, embs, index=None)
    log.info(f"[{rid}]   Cache SAVE (sans index) | dir={cache.cache_dir}")
    return (chunks, embs, index), cache, {"used_cache": False, "chunks": len(chunks)}

def _copy_for_ui(selected_images, rid: str):
    """
    Copie les images sélectionnées par le VLM dans un dossier spécifique à la requête
    pour affichage dans la galerie, sans persister le cache VLM original.
    """
    ui_dir = IMG_DIR_ROOT / "ui" / rid
    ui_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for p in selected_images or []:
        try:
            src = Path(p)
            if src.exists():
                dst = ui_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
                out_paths.append(str(dst))
        except Exception:
            continue
    return out_paths

def _clear_caches():
    import shutil as _sh
    n = 0
    for p in [Path("data/cache"), Path("data/cache_img"), IMG_DIR_ROOT]:
        if p.exists():
            _sh.rmtree(p, ignore_errors=True)
            n += 1
    for d in (Path("data/cache"), Path("data/cache_img"), IMG_DIR_ROOT):
        d.mkdir(parents=True, exist_ok=True)
    return f"Caches reset (cleared {n} directories)."

# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline (appelée par l'UI)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline (appelée par l'UI)
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_unified_ui(pdf_file, user_question, mode,
                        top_k, k_images, min_prob, page_dpi, extract_full_pages,
                        clip_model_id, qwen_vl_id):
    rid = _rid()
    if pdf_file is None or not user_question.strip():
        log.warning(f"[{rid}] Entrée incomplète (pdf/question)")
        return "Veuillez télécharger un PDF et poser une question.", "", "—", [], "Pas de diagnostic", gr.update(visible=False)

    # Device auto (pas de paramètre UI)
    text_device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[{rid}] ▶▶ Start | mode={mode} | device={text_device} | top_k={top_k} | k_img={k_images} | thr={min_prob} | dpi={page_dpi} | pages={extract_full_pages}")
    log.info(f"[{rid}]     Q: {_short(user_question)}")

    t_all0 = time.time()
    local_pdf = _ensure_uploaded(pdf_file, mode, rid)
    steps = {}
    vlm_gallery_paths = []

    try:
        # 1) Préparation texte (cache-safe: chunk params FIXES)
        t0 = time.time()
        (chunks, embs, index), _, meta = _prepare_chunks(local_pdf, mode, rid=rid)
        steps["prep_text_s"] = round(time.time() - t0, 3)
        steps["used_cache"] = meta.get("used_cache")
        steps["n_chunks"] = meta.get("chunks", 0)

        if (index is None) and embs is not None and len(embs):
            log.debug(f"[{rid}] Index None après _prepare_chunks → reconstruction")
            index = build_faiss_index(embs)
        if not chunks or index is None:
            log.warning(f"[{rid}] Aucun contenu exploitable (chunks={len(chunks) if chunks else 0}, index={'OK' if index else 'None'})")
            return "Aucun contenu exploitable trouvé dans ce PDF.", "", "—", [], "Extraction vide", gr.update(visible=False)

        # 2) Retrieval
        t0 = time.time()
        top_chunks = retrieve_top_k_chunks(user_question, chunks, embs, index, top_k=int(top_k))
        steps["retrieval_s"] = round(time.time() - t0, 3)
        log.info(f"[{rid}] ▶ Retrieval OK en {steps['retrieval_s']:.3f}s | top_k={len(top_chunks)}")

        # 3) Classification
        t0 = time.time()
        pred_class, scores = classify_prompt_zero_shot(user_question)
        steps["classify_s"] = round(time.time() - t0, 3)
        log.info(f"[{rid}] ▶ Class={pred_class} en {steps['classify_s']:.3f}s | scores={scores}")

        # 4) Génération (et détails VLM éventuels)
        force_vlm = (mode == "text-image++")
        use_pdf = local_pdf if (mode != "text-only") else None
        want_vlm_details = force_vlm and (pred_class in {"Image", "Valeur"})
        log.info(f"[{rid}] ▶ Generation | pdf_ctx={'YES' if use_pdf else 'NO'} | VLM_details={'YES' if want_vlm_details else 'NO'}")

        t0 = time.time()
        ans = generate_answer_qwen_smart(
            relevant_chunks=top_chunks,
            user_question=user_question,
            predicted_class=pred_class,
            pdf_path=use_pdf,
            force_vlm=force_vlm and pred_class in {"Image", "Valeur"},
            k_images=int(k_images),
            min_prob=float(min_prob),
            extract_full_pages=bool(extract_full_pages),
            page_dpi=int(page_dpi),
            clip_model_id=clip_model_id,
            qwen_vl_id=qwen_vl_id,
            device=text_device,               # ← on passe le device auto détecté
            return_vlm_details=want_vlm_details,
        )
        steps["generation_s"] = round(time.time() - t0, 3)
        steps["total_s"] = round(time.time() - t_all0, 3)

        if isinstance(ans, tuple):
            final_answer, vlm_meta = ans
        else:
            final_answer, vlm_meta = ans, {}

        log.info(f"[{rid}] ▶ Answer OK en {steps['generation_s']:.3f}s | total={steps['total_s']:.3f}s | len={len(str(final_answer))}")

        # 4bis) Nettoyage images + résumé & galerie (mode ++)
        vlm_summary = "—"
        if mode == "text-image++":
            try:
                pdf_hash = cache_img._pdf_hash(local_pdf)
                img_dir = IMG_DIR_ROOT / pdf_hash
                if vlm_meta:
                    vlm_answer = vlm_meta.get("vlm_answer", "").strip()
                    imgs = vlm_meta.get("selected_images", [])
                    labels = vlm_meta.get("positive_labels", [])
                    vlm_summary = (
                        "### Description VLM\n"
                        + (vlm_answer if vlm_answer else "(pas de description)") + "\n\n"
                        + ("Images sélectionnées : " + ", ".join(Path(p).name for p in imgs) + "\n" if imgs else "")
                        + ("Labels positifs : " + ", ".join(labels) + "\n" if labels else "")
                    )
                    vlm_gallery_paths = _copy_for_ui(imgs, rid)
                    steps["vlm_ui_images"] = len(vlm_gallery_paths)
                if img_dir.exists():
                    shutil.rmtree(img_dir, ignore_errors=True)
                    img_dir.mkdir(parents=True, exist_ok=True)
                steps["vlm_images_cleaned"] = True
                log.info(f"[{rid}] ▶ VLM temp images cleaned for hash={pdf_hash[:8]}")
            except Exception as _e:
                steps["vlm_images_cleaned"] = f"failed: {_e.__class__.__name__}"
                log.warning(f"[{rid}] VLM images cleanup failed: {_e}")

        # 5) Sorties + log fichier
        chunks_preview = "\n---\n".join(top_chunks)
        diag = {
            "mode": mode,
            "pred_class": pred_class,
            "scores": scores,
            "timings": steps,
            "params": {
                "top_k": int(top_k),
                "k_images": int(k_images),
                "min_prob": float(min_prob),
                "page_dpi": int(page_dpi),
                "extract_full_pages": bool(extract_full_pages),
                "clip_model_id": clip_model_id,
                "qwen_vl_id": qwen_vl_id,
                "device": text_device,        # ← consigné dans le diagnostic
            }
        }

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | rid={rid} ===\n")
            f.write(f"PDF: {local_pdf}\nMode: {mode}\nQuestion: {user_question}\n")
            f.write(f"Classe: {pred_class} | Scores: {scores}\n")
            f.write(f"Timings: {json.dumps(steps)}\n")
            f.write(f"Params: {json.dumps(diag['params'])}\n")
            f.write("Top chunks:\n" + "\n\n".join(top_chunks) + "\n")
            f.write("Réponse:\n" + str(final_answer).strip() + "\n")
            if vlm_summary and vlm_summary != "—":
                f.write("Résumé VLM:\n" + vlm_summary + "\n")

        return str(final_answer).strip(), chunks_preview.strip(), vlm_summary, vlm_gallery_paths, json.dumps(diag, ensure_ascii=False, indent=2), gr.update(visible=True, value=str(LOG_FILE))

    except Exception as e:
        log.exception(f"[{rid}] ❌ Pipeline error: {type(e).__name__}: {e}")
        return f"Erreur : {type(e).__name__} — {str(e)}", "", "—", [], "Erreur (voir logs)", gr.update(visible=True, value=str(LOG_FILE))

# ─────────────────────────────────────────────────────────────────────────────
# UI Gradio (complète + scroll + About + Galerie)
# ─────────────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Qwen PDF Q&A — Unified", theme=gr.themes.Soft(), fill_height=True) as demo:
        # CSS pour rendre les zones défilables
        gr.HTML("""
        <style>
          #answer_box textarea, #passages_box textarea, #vlm_box textarea {
            height: 56vh !important;
            max-height: 70vh;
            overflow: auto !important;
            resize: vertical;
            white-space: pre-wrap;
          }
          .gr-gallery { max-height: 60vh; overflow: auto; }
          @media (max-width: 900px) {
            #answer_box textarea, #passages_box textarea, #vlm_box textarea { height: 44vh !important; }
          }
        </style>
        """)

        with gr.Tabs():
            # ───────── Tab: Ask ─────────
            with gr.Tab("Ask"):
                gr.Markdown("# Qwen PDF Q&A — Unified\nAnalyse multimodale (texte + images) avec cache maîtrisé.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=380):
                        pdf_in   = gr.File(label="PDF à analyser", file_types=[".pdf"])
                        question = gr.Textbox(label="Question", placeholder="Ex : Quel est le chiffre d'affaires du segment E ?", lines=2)
                        mode     = gr.Radio(choices=MODES, value="text-only", label="Mode")
                        run_btn  = gr.Button("Analyser", variant="primary")
                    with gr.Column(scale=1):
                        answer_out   = gr.Textbox(label="Réponse générée", lines=12, elem_id="answer_box")
                        passages_out = gr.Textbox(label="Passages extraits (top-k)", lines=12, elem_id="passages_box")
                with gr.Row():
                    with gr.Column(scale=1):
                        vlm_out = gr.Textbox(label="Résumé VLM (mode ++)", lines=12, elem_id="vlm_box")
                    with gr.Column(scale=1):
                        vlm_gallery = gr.Gallery(label="Images VLM sélectionnées (mode ++)", columns=4, show_label=True)

            # ───────── Tab: Paramètres (sans impact cache) ─────────
            with gr.Tab("Paramètres (sans impact cache)"):
                gr.Markdown("Ces réglages **n’altèrent pas** les chunks/embeddings stockés.")
                with gr.Row():
                    with gr.Column():
                        top_k          = gr.Slider(1, 10, step=1, value=DEF_TOP_K, label="Top-k passages (retrieval)")
                    with gr.Column():
                        k_images       = gr.Slider(1, 8, step=1, value=DEF_K_IMAGES, label="VLM: nombre d’images (k_images)")
                        min_prob       = gr.Slider(0.05, 0.9, step=0.01, value=DEF_MIN_CLIP_PROB, label="VLM: seuil CLIP (min_prob)")
                    with gr.Column():
                        page_dpi       = gr.Slider(72, 300, step=12, value=DEF_PAGE_DPI, label="VLM: rendu de page (DPI)")
                        extract_full_pages = gr.Checkbox(DEF_EXTRACT_PAGES, label="VLM: extraire rendu de pages (vectoriels)")
                with gr.Row():
                    clip_model_id   = gr.Textbox(value=DEF_CLIP_MODEL_ID, label="VLM: CLIP model id")
                    qwen_vl_id      = gr.Textbox(value=DEF_QWEN_VL_ID, label="VLM: Qwen-VL model id")
                    qwen_embed      = gr.Textbox(value=DEF_QWEN_EMBED, label="EMBEDDING: Qwen model id")
                    qwen_llm        = gr.Textbox(value=DEF_QWEN_LLM, label="LLM: Qwen model id")


            # ───────── Tab: About ─────────
            with gr.Tab("About"):
                gr.Markdown("""
### À propos
**Qwen PDF Q&A — Unified** combine :
- **Docling** (texte, tableaux, descriptions d’images),
- **Chunking** (préservation des tableaux, overlap),
- **Embeddings** Qwen3 + recherche (NearestNeighbors) + **CrossEncoder** pour le reranking,
- **VLM Qwen2.5-VL** pour *Image/Valeur* en mode **text-image++**,
- **Fusion** texte + résumé visuel pour une réponse unique.

### Modes
- **text-only** : pas d’analyse d’images.
- **text-image-with-docling** : descriptions d’images par Docling, pas de VLM.
- **text-image++** : VLM (sélection via CLIP, description via Qwen-VL) déclenché si la question est classée *Image* ou *Valeur*.

### Cache
- **Texte** : `data/cache` (text-only) et `data/cache_img` (text-image/++).
- **Images VLM** : non persistées (nettoyées après exécution **++**).  
  Les vignettes UI sont copiées dans `data/images/ui/<rid>/` et supprimées par “Vider les caches”.

### Conseils
- Pour graphiques vectoriels, cochez “extraire rendu de pages” et/ou augmentez le DPI.
- Si la réponse est courte, regardez le **Résumé VLM** et la **galerie** : cela révèle ce que le modèle a effectivement vu.
""")

            # ───────── Tab: Diagnostic & Log ─────────
            with gr.Tab("Diagnostic & Log"):
                with gr.Row():
                    diag_out  = gr.Code(label="Diagnostic (JSON)", language="json")
                    log_dl    = gr.File(label="Télécharger le log d'exécution", visible=False)
                with gr.Row():
                    clear_btn = gr.Button("Vider les caches (texte / image)")
                    clear_info = gr.Markdown("")

        # actions
        run_btn.click(
            fn=pipeline_unified_ui,
            inputs=[pdf_in, question, mode,
                    top_k, k_images, min_prob, page_dpi, extract_full_pages,
                    clip_model_id, qwen_vl_id],
            outputs=[answer_out, passages_out, vlm_out, vlm_gallery, diag_out, log_dl]
        )

        def _do_clear():
            return _clear_caches()
        clear_btn.click(fn=_do_clear, outputs=[clear_info])

    return demo

demo = build_ui()

if __name__ == "__main__":
    # Pour plus de logs: export LOGLEVEL=DEBUG
    demo.launch()
