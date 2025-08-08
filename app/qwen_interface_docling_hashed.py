# interface_gradio.py

import time, shutil
from pathlib import Path
import gradio as gr

from pipeline.extract.extract_with_docling import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format
from utils.cache_manager import compute_file_hash, get_cached_paths, is_cached

# Dossiers de travail pour les fichiers temporaires
PROJECT_ROOT = Path("data")
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
MD_DIR       = PROJECT_ROOT / "markdown"
CHUNK_DIR    = PROJECT_ROOT / "chunks"
LOG_FILE     = PROJECT_ROOT / "logs" / "interface_log.txt"

for d in (UPLOAD_DIR, MD_DIR, CHUNK_DIR, LOG_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)

def pipeline_question_answer(pdf_file, user_question, top_k=3):
    if pdf_file is None or not user_question.strip():
        return "Veuillez télécharger un PDF et poser une question.", ""

    ts = int(time.time() * 1000)
    local_pdf = UPLOAD_DIR / f"upload_{ts}_{Path(pdf_file.name).stem}.pdf"
    shutil.copy(pdf_file.name, local_pdf)

    try:
        # Vérification cache
        pdf_hash = compute_file_hash(local_pdf)
        cached = get_cached_paths(pdf_hash, UPLOAD_DIR)

        if not is_cached(cached):
            cached["pdf"].write_bytes(local_pdf.read_bytes())
            md_path = pdf_to_markdown(local_pdf, output_dir=MD_DIR)
            chunk_markdown(md_path, chunk_size=2048, output_chunks_path=cached["chunks"])

        text_chunks = cached["chunks"].read_text(encoding="utf-8").split("\n---\n")
        if not text_chunks:
            return "Aucun contenu exploitable trouvé dans ce PDF.", ""

        embeddings = embed_chunks_qwen3(text_chunks)
        index = build_faiss_index(embeddings)
        top_chunks = retrieve_top_k_chunks(user_question, text_chunks, embeddings, index, top_k=top_k)

        answer = generate_answer_qwen_chat_format(top_chunks, user_question)
        chunks_preview = "\n---\n".join(top_chunks)

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n\n=== PDF ===  {local_pdf}")
            f.write(f"\n=== HASH  ===  {pdf_hash}")
            f.write(f"\n=== Question ===\n{user_question}\n")
            f.write("\n=== Chunks retenus ===\n" + "\n\n".join(top_chunks) + "\n")
            f.write("\n=== Réponse générée ===\n" + answer + "\n")

        return answer.strip(), chunks_preview.strip()

    except Exception as e:
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
