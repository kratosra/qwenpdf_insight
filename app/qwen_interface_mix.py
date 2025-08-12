# app/qwen_interface_docling_img.py  (fixed)
from pathlib import Path
import gradio as gr
from pipeline.multimodal.multimodal_orchestrator import answer_multimodal

def qa_fusion(pdf_file, question, k_txt, k_img, p_min, full_pages):
    if pdf_file is None or not str(question).strip():
        return "Télécharge un PDF et pose une question.", "", "", []
    out = answer_multimodal(
        pdf_path=Path(pdf_file.name),
        user_question=str(question),
        top_k_chunks=int(k_txt),
        k_images=int(k_img),
        min_prob=float(p_min),
        extract_full_pages=bool(full_pages),
    )
    return (
        out["final_answer"],
        "\n\n---\n".join(out["text_top_chunks"]),
        out["vlm_answer"],
        out["selected_images"],
    )

with gr.Blocks(title="QwenPDF Insight") as demo:
    with gr.Tabs():
        with gr.TabItem("Fusion Texte + Images"):
            pdf = gr.File(label="PDF", file_types=[".pdf"])
            q   = gr.Textbox(label="Question")
            with gr.Row():
                k_txt = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top-k texte")
                k_img = gr.Slider(minimum=1, maximum=6, value=4, step=1, label="Top-k images")
            with gr.Row():
                pmin  = gr.Slider(minimum=0.05, maximum=0.5, value=0.18, step=0.01, label="Seuil proba min (zero-shot)")
                fullp = gr.Checkbox(value=False, label="Rendre pages entières (vecteurs/graphs)")
            run = gr.Button("Analyser (fusion)")
            final      = gr.Textbox(label="Réponse finale", lines=10)
            chunks_box = gr.Textbox(label="Extraits texte retenus", lines=10)
            vlm_box    = gr.Textbox(label="Résumé visuel (VLM)", lines=10)
            gallery    = gr.Gallery(label="Images utilisées", columns=4, height=320, preview=True)

            run.click(
                qa_fusion,
                [pdf, q, k_txt, k_img, pmin, fullp],
                [final, chunks_box, vlm_box, gallery],
            )

# launch_gradio_mix.py imports `demo` from this module, so no __main__ needed here.
