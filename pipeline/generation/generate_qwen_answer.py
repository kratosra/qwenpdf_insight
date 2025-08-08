# generate_qwen_answer.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# Chargement du modèle Qwen Chat
chat_model_id = "Qwen/Qwen3-0.6B"
tokenizer_chat = AutoTokenizer.from_pretrained(chat_model_id, trust_remote_code=True)
model_chat = AutoModelForCausalLM.from_pretrained(
    chat_model_id,
    torch_dtype="auto",
    device_map="auto"
)
model_chat.eval()

# Initialisation du classifieur zero-shot
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
candidate_labels = ["Résumé", "Valeur", "Image", "Autre"]

# Prompts spécialisés par classe
prompt_templates = {
    "Résumé": (
        "Tu es un expert en analyse de documents. À partir des extraits textuels suivants issus d’un document PDF, génère une synthèse claire, informative et factuelle.\n"
        " Règles importantes :\n"
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
    "Ta mission est d’identifier avec précision toutes les informations pertinentes relatives à la valeur mentionnée dans la question, "
    "en t’appuyant exclusivement sur les extraits fournis, qu’ils soient issus du texte ou de tableaux.\n\n"
    
    "Si la valeur est explicitement indiquée, reproduis-la fidèlement. Si elle figure dans un tableau ou dans une phrase complexe, reformule-la de manière claire et compréhensible. "
    "En cas de présence de plusieurs valeurs liées, présente-les toutes en précisant leur contexte (année, catégorie, segment, etc.).\n\n"

    "N’ajoute aucune supposition : si l’information est incomplète, ambigüe ou absente, indique-le honnêtement.\n\n"

    "Lorsque tu cites une valeur, précise toujours sa provenance en incluant un extrait représentatif (phrase ou cellule de tableau).\n\n"

    "Extraits fournis :\n{context}\n\n"
    "Question : {query}\n"
    "Réponse :"
)

,
    "Image": (
        "Tu es un analyste financier expert. Analyse UNIQUEMENT ce que tu vois dans l'image ou la description d’image extraite du document financier ci-joint.\n"
        "Important : NE FAIS PAS D'HYPOTHÈSES. Si des données ne sont pas claires ou manquent, indique-le honnêtement.\n\n"
        "Réponds précisément à cette question : {query}\n\n"
        "Structure ta réponse ainsi :\n"
        "1. Décris exactement les éléments visuels présents (graphique, tableau, schéma...)\n"
        "2. Liste les valeurs exactes que tu peux lire ou identifier (avec unités si présentes)\n"
        "3. Si tu vois des tendances ou variations (%), indique-les avec précision\n"
        "4. Si des informations nécessaires sont absentes ou illisibles, signale-le clairement\n\n"
        "Description du document  :\n{context}\n"
        "Réponse :"

    ),
    "Autre": (
        "Tu es un assistant intelligent spécialisé dans l’analyse de documents. Réponds au prompt posée uniquement à partir des extraits suivants.\n"
        " Ne fais aucune supposition. Ne t'appuie que sur les contenus fournis.\n"
        "Si l'information demandée n’est pas disponible, indique-le clairement.\n\n"
        "Extraits fournis :\n{context}\n\n"
        "Question : {query}\n"
        "Réponse :"
    )
}

def classify_prompt_zero_shot(prompt: str):
    """
    Utilise un modèle zero-shot pour classifier un prompt selon les 4 catégories.
    Retourne la classe prédite et les scores de confiance.
    """
    result = classifier(prompt, candidate_labels)
    pred_class = result["labels"][0]
    scores = {label: round(score, 3) for label, score in zip(result["labels"], result["scores"])}
    return pred_class, scores
"""
def verify_answer(answer: str, context: str, threshold: float = 0.5) -> bool:
    answer_tokens = re.findall(r"\w+", answer.lower())
    context_tokens = set(re.findall(r"\w+", context.lower()))
    if not answer_tokens:
        return False
    match_count = sum(1 for t in answer_tokens if t in context_tokens)
    return (match_count / len(answer_tokens)) >= threshold
"""
def generate_answer_qwen_chat_format(relevant_chunks, user_question, predicted_class=None, device: str = "cuda"):
    context = "\n\n".join(relevant_chunks)

    # Classification automatique si la classe n'est pas fournie
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
    inputs = tokenizer_chat([text_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_chat.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            min_p=0,
            temperature=0.6,
            eos_token_id=tokenizer_chat.eos_token_id,
            pad_token_id=tokenizer_chat.eos_token_id
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer_chat.decode(generated, skip_special_tokens=True).strip()

    return decoded
