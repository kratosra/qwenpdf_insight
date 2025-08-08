from transformers import pipeline
import json

# Charger le pipeline zero-shot
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# Liste des classes cibles
candidate_labels = ["Résumé", "Valeur", "Image", "Autre"]

test_prompts = [
    #  Synthèse / Résumé
    "Fais-moi un résumé de ce rapport.",
    "Peux-tu synthétiser le contenu du document ?",
    "Résume les points clés du fichier PDF.",
    "Donne les principales conclusions de ce document.",
    "Je veux une vue d'ensemble rapide.",

    #  Valeurs / KPIs / Chiffres
    "Quelle est la valeur du chiffre d'affaires en 2023 ?",
    "Quel est le bénéfice net en 2024 ?",
    "Combien d'unités ont été vendues ?",
    "Donne-moi les indicateurs de performance.",
    "Quel est le revenu annuel pour chaque segment ?",
    "Quels sont les chiffres clés du rapport ?",
    "Quel est le montant total du budget ?",

    #  Images / Graphiques / Visuels
    "Montre-moi les graphiques du rapport.",
    "Quels tableaux sont mentionnés à la page 4 ?",
    "Affiche les schémas présents dans le fichier.",
    "Quels sont les graphiques disponibles ?",
    "Montre les illustrations du document.",
    "Quels visuels montrent l’évolution annuelle ?",
    "Y a-t-il des infographies dans le rapport ?",
    "Donne-moi tous les éléments visuels.",

    #  Structure / Sections / Plan
    "Décris la structure du rapport.",
    "Quel est le plan du document ?",
    "Décris-moi la structure du fichier.",
    "Combien de sections contient le rapport ?",
    "Y a-t-il une table des matières ?",
    "Quelles sont les différentes parties du document ?",

    #  Question / Recherche libre
    "Quel est le nom du directeur ?",
    "Qui est le PDG de l’entreprise ?",
    "Quels sont les partenaires stratégiques ?",
    "Où se trouve le siège social ?",
    "Quels projets sont mentionnés ?",
    "À quoi sert cette plateforme ?",

    #  Autre / Ambigu / Non-structuré
    "Je veux tout savoir.",
    "Et pour la suite ?",
    "Qu’en est-il de l’année prochaine ?",
    "Tu peux me dire plus ?",
    "Analyse complète stp."
]
# Lancer la classification
for prompt in test_prompts:
    result = classifier(prompt, candidate_labels)
    top_label = result['labels'][0]
    scores = {label: round(score, 3) for label, score in zip(result['labels'], result['scores'])}

    print(json.dumps({
        "prompt": prompt,
        "predicted_class": top_label,
        "scores": scores
    }, indent=2, ensure_ascii=False))
    print("=" * 80)
