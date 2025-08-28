# pipeline/generation/generate_qwen_answer.py

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import gc
import re

# VLM pipeline (sélection d’images + Qwen-VL)
#from pipeline.multimodal.image_qa_pipeline_quantized import multimodal_image_qa_pipeline
from pipeline.multimodal.image_qa_pipeline_quantized import multimodal_image_qa_pipeline

# ----------------------------
# Utilitaire : libérer la VRAM
# ----------------------------
def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------
# Modèle Qwen Chat (texte) - 4-bit pour réduire la VRAM
# ----------------------------
chat_model_id = "Qwen/Qwen3-0.6B"
tokenizer_chat = AutoTokenizer.from_pretrained(chat_model_id, trust_remote_code=True)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # si erreur bfloat16 -> passer à torch.float16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_chat = AutoModelForCausalLM.from_pretrained(
    chat_model_id,
    device_map="auto",              # on laisse Accelerate dispatcher sur GPU/CPU
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)
model_chat.eval()

# ----------------------------
# Classifieur zero-shot (CPU)
# ----------------------------
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=-1  # CPU pour ne pas prendre de VRAM
)
candidate_labels = ["Résumé", "Valeur", "Image", "Autre"]

# ----------------------------
# Prompts par classe
# ----------------------------
prompt_templates = {
    "Résumé": (
        "Tu es un expert en analyse de documents. À partir des extraits textuels suivants issus d’un document PDF, "
        "génère une synthèse claire, informative et factuelle.\n"
        "Règles importantes :\n"
        "• Ne résume que ce qui est dans les extraits fournis, ne complète pas avec des connaissances extérieures.\n"
        "• Évite les interprétations subjectives.\n"
        "• Conserve les termes techniques utilisés dans le texte.\n\n"
        "Structure ta réponse ainsi :\n"
        "1. Contexte général du document (si identifiable)\n"
        "2. Points clés abordés dans les extraits\n"
        "3. Données ou conclusions importantes citées\n\n"
        "Extraits fournis :\n{context}\n\n"
        "Demande : {query}\n"
        "Réponse :"
    ),
    "Valeur": (
        "Tu es un assistant expert en analyse de documents financiers et en extraction de données chiffrées. "
        "Ta mission est d’identifier avec précision toutes les informations pertinentes relatives à la valeur mentionnée "
        "dans la question, en t’appuyant exclusivement sur les extraits fournis (texte ou tableaux).\n\n"
        "Si la valeur est explicitement indiquée, reproduis-la fidèlement. Si elle figure dans un tableau ou une phrase "
        "complexe, reformule-la clairement. En cas de plusieurs valeurs liées, liste-les avec leur contexte (année, segment…).\n\n"
        "N’ajoute aucune supposition. Si l’info est absente/ambigüe, dis-le.\n\n"
        "Inclue, quand c’est utile, un court extrait (phrase/cellule) d’où provient la donnée.\n\n"
        "Extraits fournis :\n{context}\n\n"
        "Question : {query}\n"
        "Réponse :"
    ),
    "Image": (
        "Tu es un analyste financier. Analyse UNIQUEMENT ce que tu vois dans l'image ou la description d’image.\n"
        "Ne fais aucune hypothèse. Si certaines valeurs manquent, dis-le.\n\n"
        "Question : {query}\n\n"
        "Réponds en 4 points :\n"
        "1) Éléments visuels présents (graphique/tableau…)\n"
        "2) Valeurs lisibles (avec unités)\n"
        "3) Tendances/variations (%) si visibles\n"
        "4) Manques/incertitudes\n\n"
        "Description :\n{context}\n\n"
        "Réponse :"
    ),
    "Autre": (
        "Tu es un assistant spécialisé en analyse de documents. Réponds uniquement à partir des extraits fournis. "
        "Ne fais aucune supposition. Si l'information demandée n’est pas disponible, indique-le clairement.\n\n"
        "Extraits fournis :\n{context}\n\n"
        "Question : {query}\n"
        "Réponse :"
    ),
}

# ----------------------------
# Helpers
# ----------------------------
def classify_prompt_zero_shot(prompt: str):
    """Retourne (classe prédite, scores) pour le prompt utilisateur."""
    result = classifier(prompt, candidate_labels)
    pred_class = result["labels"][0]
    scores = {label: round(score, 3) for label, score in zip(result["labels"], result["scores"])}
    return pred_class, scores

def _fusion_prompt(text_chunks: List[str], vlm_answer: str, question: str) -> str:
    text_block = "\n\n---\n".join(text_chunks) if text_chunks else "(aucun extrait textuel pertinent trouvé)"
    visual_block = vlm_answer.strip() if vlm_answer else "(aucun élément visuel pertinent trouvé)"
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
def generate_answer_qwen_chat_format(
    relevant_chunks: List[str],
    user_question: str,
    predicted_class: Optional[str] = None,
    device: Optional[torch.device] = None,   # laissé pour compat
):
    import gc

    def _flush_vram():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    context = "\n\n".join(relevant_chunks)

    if predicted_class is None:
        predicted_class, _ = classify_prompt_zero_shot(user_question)

    template = prompt_templates.get(predicted_class, prompt_templates["Résumé"])
    prompt = template.format(context=context, query=user_question)

    messages = [
        {"role": "system", "content": "Tu es un assistant rigoureux et réponds en français. Ne fais aucune supposition."},
        {"role": "user",   "content": prompt}
    ]

    text_prompt = tokenizer_chat.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    # Tokenisation (CPU par défaut)
    inputs = tokenizer_chat([text_prompt], return_tensors="pt")

    #  Aligner les entrées sur le device des embeddings du modèle (robuste même avec device_map="auto")
    try:
        model_device = model_chat.get_input_embeddings().weight.device
    except Exception:
        try:
            model_device = next(model_chat.parameters()).device
        except Exception:
            model_device = torch.device("cpu")

    if hasattr(inputs, "to"):
        inputs = inputs.to(model_device)
    else:
        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Génération (ne *pas* faire model_chat.to(...) quand device_map="auto")
    with torch.no_grad():
        outputs = model_chat.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=1024,   # plus safe côté VRAM
            do_sample=True,
            top_p=0.95,
            top_k=20,
            temperature=0.6,
            eos_token_id=tokenizer_chat.eos_token_id,
            pad_token_id=tokenizer_chat.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer_chat.decode(generated, skip_special_tokens=True).strip()

    #  Vider la VRAM après génération
    _flush_vram()

    return decoded


# ----------------------------
# Fonction "smart" (texte seul ou fusion texte+VLM)
# ----------------------------
def generate_answer_qwen_smart(
    relevant_chunks: List[str],
    user_question: str,
    predicted_class: Optional[str] = None,
    pdf_path: Optional[str | Path] = None,
    # options VLM
    k_images: int = 4,
    min_prob: float = 0.18,
    extract_full_pages: bool = False,
    page_dpi: int = 180,
    clip_model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    qwen_vl_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = None,  # conservé pour compat, non utilisé
    force_vlm: bool = False,
    return_vlm_details: bool = False,  # ← retourne (réponse, meta VLM) si True
):
    """
    Génère une réponse finale.
    - Si (classe ∈ {Image, Valeur} ou force_vlm=True) ET pdf_path fourni → lance VLM,
      fusionne texte+visuel, et génère la réponse finale texte.
    - Sinon → génération texte standard.
    - Si return_vlm_details=True, renvoie (final_answer, vlm_meta_dict) ; sinon une string.
    """

    # 1) Classe auto si absente
    if predicted_class is None:
        predicted_class, _ = classify_prompt_zero_shot(user_question)

    need_vlm = force_vlm or (predicted_class in {"Image", "Valeur"})
    vlm_answer = ""
    vlm_meta: Dict[str, Any] = {}

    # 2) Analyse visuelle si requise
    if need_vlm and pdf_path is not None:
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
        vlm_meta = {
            "vlm_answer": vlm_answer,
            "selected_images": img_out.get("selected_images", []),
            "positive_labels": img_out.get("positive_labels", []),
            "image_dir": img_out.get("image_dir", ""),
            "zs_scores": img_out.get("zs_scores", {}),
        }

        # libérer la VRAM avant la génération texte
        _free_cuda()

        fusion = _fusion_prompt(relevant_chunks, vlm_answer, user_question)
        final = generate_answer_qwen_chat_format([fusion], user_question, predicted_class="Autre")
        return (final, vlm_meta) if return_vlm_details else final

    # 3) Sinon : texte standard
    _free_cuda()
    final = generate_answer_qwen_chat_format(relevant_chunks, user_question, predicted_class=predicted_class)
    return (final, {}) if return_vlm_details else final
