import time, shutil, logging, os
from pathlib import Path
import gradio as gr

from pipeline.cache_manager import EmbeddingCacheManager
from pipeline.extract.extract_with_docling import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path("data")
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
MD_DIR       = PROJECT_ROOT / "markdown"
CHUNK_DIR    = PROJECT_ROOT / "chunks"
LOG_FILE     = PROJECT_ROOT / "logs" / "interface_log.txt"
for d in (UPLOAD_DIR, MD_DIR, CHUNK_DIR, LOG_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)

cache_manager = EmbeddingCacheManager(cache_dir="data/cache")

def pipeline_question_answer(pdf_file, user_question, top_k=3):
    if pdf_file is None or not user_question.strip():
        return "Veuillez télécharger un PDF et poser une question.", ""

    # 1) Nom d’upload déterministe = hash → pas de doublons
    src = Path(pdf_file.name)
    file_hash = cache_manager._pdf_hash(src)   # OK on réutilise la même fonction
    local_pdf = UPLOAD_DIR / f"{file_hash}.pdf"

    if local_pdf.exists():
        log.info(f"Upload skipped (duplicate): {local_pdf.name}")
    else:
        shutil.copy(src, local_pdf)
        log.info(f"Uploaded new file: {pdf_file.name}")

    try:
        start_time = time.time()

        # 2) Cache embeddings/index
        cached = cache_manager.load_cache(local_pdf)
        if cached:
            text_chunks, embeddings, index = cached
            log.info(f"Loaded cached IMG embeddings/index for {local_pdf.name} in {time.time() - start_time:.2f} sec.")
        else:
            log.info(f"Processing new IMG document {local_pdf.name} ...")
            md_path = pdf_to_markdown(local_pdf, output_dir=MD_DIR)
            chunk_file = CHUNK_DIR / f"{md_path.stem}_chunks.txt"
            text_chunks = chunk_markdown(md_path, chunk_size=2048, output_chunks_path=chunk_file)

            if not text_chunks:
                return "Aucun contenu exploitable trouvé dans ce PDF.", ""

            embeddings = embed_chunks_qwen3(text_chunks)
            index = build_faiss_index(embeddings)

            cache_manager.save_cache(local_pdf, text_chunks, embeddings, index)
            log.info(f"Finished converting IMG document {local_pdf.name} in {time.time() - start_time:.2f} sec.")

        # 3) Recherche + génération
        top_chunks = retrieve_top_k_chunks(user_question, text_chunks, embeddings, index, top_k=top_k)
        answer = generate_answer_qwen_chat_format(top_chunks, user_question)
        chunks_preview = "\n---\n".join(top_chunks)

        # 4) Log fichier
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n\n=== PDF ===  {local_pdf}")
            f.write(f"\n=== Question ===\n{user_question}\n")
            f.write("\n=== Chunks retenus ===\n" + "\n\n".join(top_chunks) + "\n")
            f.write("\n=== Réponse générée ===\n" + answer + "\n")

        return answer.strip(), chunks_preview.strip()

    except Exception as e:
        log.error(f"Error processing {local_pdf.name}: {type(e).__name__} — {str(e)}")
        return f"Erreur : {type(e).__name__} — {str(e)}", ""

demo = gr.Interface(
    fn=pipeline_question_answer,
    inputs=[
        gr.File(label="PDF à analyser", file_types=[".pdf"]),
        gr.Textbox(label="Question", placeholder="Ex : Quel est le chiffre d'affaires du segment E ?"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Nombre de passages (top-k)"),
    ],
    outputs=[
        gr.Textbox(label="Réponse générée"),
        gr.Textbox(label="Passages extraits (top-k)", lines=12),
    ],
    title="Qwen PDF Q&A (Docling-powered)",
    description="Chargez un PDF comportant texte et/ou tableaux, posez votre question, puis obtenez la réponse accompagnée des passages sources.",
)

if __name__ == "__main__":
    demo.launch()
