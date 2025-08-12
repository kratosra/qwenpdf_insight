# pipeline/multimodal/multimodal_orchestrator.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- texte (déjà dans ton repo)
from pipeline.extract.extract_with_docling_img import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format

# --- cache embeddings/index (fourni par toi)
from pipeline.cache_manager import EmbeddingCacheManager

# --- images (nouveau module déjà fourni)
from pipeline.multimodal.image_qa_pipeline_quantized import multimodal_image_qa_pipeline

def _ensure_text_index_with_cache(
    pdf_path: Path,
    chunk_size: int = 2048
) -> Tuple[List[str], List[float], object]:
    """
    Retourne (chunks, embeddings, index). Utilise EmbeddingCacheManager.
    Si cache manquant -> docling->chunks->embed->index, puis save_cache.
    """
    cache = EmbeddingCacheManager(cache_dir="data/cache")
    cached = cache.load_cache(pdf_path)
    if cached:
        return cached  # (chunks, embeddings, index)

    # Pas de cache → pipeline texte
    md_path = pdf_to_markdown(pdf_path, output_dir="data/markdown")
    chunks = chunk_markdown(md_path, chunk_size=chunk_size, output_chunks_path=None)

    if not chunks:
        # On garde un index vide compatible
        return [], [], None

    embeddings = embed_chunks_qwen3(chunks)
    index = build_faiss_index(embeddings)
    cache.save_cache(pdf_path, chunks, embeddings, index)
    return chunks, embeddings, index


def _final_fusion_prompt(text_evidence: List[str], visual_summary: str, question: str) -> str:
    """
    Construit un prompt de synthèse final pour Qwen (texte) qui :
    - rappelle la question,
    - donne les extraits textuels retenus,
    - donne le résumé visuel issu du VLM,
    - demande une réponse précise & sourcée,
    - interdit les suppositions.
    """
    text_block = "\n\n---\n".join(text_evidence) if text_evidence else "(aucun extrait textuel pertinent trouvé)"
    visual_block = visual_summary.strip() if visual_summary else "(aucun élément visuel pertinent trouvé)"
    return (
        "Tu es un analyste rigoureux. Fourni une réponse précise et factuelle à partir des SEULES informations ci-dessous.\n"
        "N'invente rien. Si une donnée manque, dis-le clairement.\n\n"
        f"Question : {question}\n\n"
        "Éléments TEXTUELS retenus (top-k) :\n"
        f"{text_block}\n\n"
        "Éléments VISUELS (résumé ciblé du VLM) :\n"
        f"{visual_block}\n\n"
        "Consigne : réponds de façon concise, cite explicitement les valeurs (unités, périodes), et précise si elles viennent du texte ou d’un visuel. "
        "S’il existe des écarts entre texte et visuel, signale-les. Réponse :"
    )


def answer_multimodal(
    pdf_path: str | Path,
    user_question: str,
    top_k_chunks: int = 3,
    chunk_size: int = 2048,
    k_images: int = 4,
    min_prob: float = 0.18,
    extract_full_pages: bool = False,
    page_dpi: int = 180,
    clip_model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    qwen_vl_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda"
) -> Dict[str, object]:
    """
    Orchestration complète :
      1) Texte : cache→chunks→embeddings→index→retrieve top-k
      2) Images : zero-shot CLIP→sélection→Qwen-VL
      3) Fusion : prompt final (texte) avec les deux sources → réponse unique
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    # 1) TEXTE — cache + retrieval
    chunks, embeddings, index = _ensure_text_index_with_cache(pdf_path, chunk_size=chunk_size)
    top_chunks: List[str] = []
    if chunks and embeddings is not None and index is not None:
        top_chunks = retrieve_top_k_chunks(user_question, chunks, embeddings, index, top_k=top_k_chunks)

    # 2) IMAGES — zero-shot + Qwen-VL
    img_out = multimodal_image_qa_pipeline(
        pdf_path=pdf_path,
        user_question=user_question,
        images_root="data/images",
        k_images=k_images,
        min_prob=min_prob,
        clip_model_id=clip_model_id,
        qwen_vl_id=qwen_vl_id,
        extract_full_pages=extract_full_pages,
        page_dpi=page_dpi,
    )
    vlm_answer = (img_out.get("answer") or "").strip()

    # 3) FUSION — une réponse finale via le générateur texte
    fusion_prompt = _final_fusion_prompt(top_chunks, vlm_answer, user_question)
    # On réutilise ton générateur Qwen texte — on lui passe la "classe" générique Autre
    final_answer = generate_answer_qwen_chat_format(
        relevant_chunks=[fusion_prompt],  # on met tout le contexte fusionné dans un seul "chunk"
        user_question=user_question,
        predicted_class="Autre",
        device=device
    )

    return {
        "final_answer": final_answer,
        "text_top_chunks": top_chunks,
        "vlm_answer": vlm_answer,
        "selected_images": img_out.get("selected_images", []),
        "zs_scores": img_out.get("zs_scores", {}),
        "positive_labels": img_out.get("positive_labels", []),
        "image_dir": img_out.get("image_dir", ""),
    }
