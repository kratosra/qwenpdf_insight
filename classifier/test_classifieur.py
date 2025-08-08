from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import json

# Charger modèle et tokenizer
model_path = "./results_classifier/checkpoint-8433"
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

# Labels utilisés à l'entraînement (6 classes)
id2label = {
    0: "Image",
    1: "Résumé",
    2: "Question",
    3: "Autre",
    5: "Valeur",
    4: "Structure"
}

# Fonction de prédiction
def predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_class = id2label[int(torch.argmax(logits))]
    scores = {id2label[i]: round(p, 3) for i, p in enumerate(probs)}
    return {"prompt": prompt, "predicted_class": pred_class, "scores": scores}


# Exemple d'utilisation
if __name__ == "__main__":
    test_prompts = [
        "Donne-moi une synthèse du document.",
        "Quelle est la valeur du chiffre d'affaires en 2023 ?",
        "Montre-moi les graphiques du rapport.",
        "Quels tableaux sont mentionnés à la page 4 ?",
        "Décris la structure du rapport.",
        "Quel est le nom du directeur ?",
        "Donne-moi tous les éléments visuels.",
        "Quel est le montant total du budget ?",
        "De quoi il s'agit ce document ?",
        "Quels sont les principaux résultats présentés ?",
        "Quel est le délai de livraison mentionné ?",
        "Quels sont les risques identifiés dans le rapport ?",
        "Quelle est la date de la réunion ?",
        "Quels sont les objectifs du projet ?",
        "Quels sont les indicateurs de performance clés ?",
        "Quel est le budget alloué pour cette initiative ?",
        "Quels sont les partenaires impliqués ?",
        "Quels sont les principaux défis rencontrés ?",
        "Quels sont les résultats financiers de l'année précédente ?",
        "Quels sont les principaux enseignements tirés ?"
    ]

    for prompt in test_prompts:
        result = predict(prompt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 80)