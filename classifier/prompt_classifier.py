from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

prompts = [
    "Donne-moi une synthèse du document.",
    "Quelle est la valeur du chiffre d'affaires en 2023 ?",
    "Montre-moi les graphiques du rapport.",
    "Quels tableaux sont mentionnés à la page 4 ?",
    "Décris la structure du rapport.",
    "Quel est le nom du directeur ?",
    "Donne-moi tous les éléments visuels.",
    "De quoi il s'agit ce document ?"
]

labels = ["Résumé", "Image","Valeur", "Autre"]

for p in prompts:
    result = classifier(p, labels, multi_label=False, hypothesis_template ="Dans le cadre d'une conversation sur un PDF, cette question vise : {}."

)
    print(f"\nPrompt : {p}")
    print("→ Classe prédite :", result['labels'][0])
    print("→ Scores :", {l: round(s, 3) for l, s in zip(result['labels'], result['scores'])})
