# image_zero_shot.py
import os
from typing import Dict, List, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# === Modèle OpenCLIP (pas OpenAI/Google) ===
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Définitions de classes et prompts multi-lingues (FR/EN) + synonymes
CLASS_PROMPTS: Dict[str, List[str]] = {
    "graphique": [
        "un graphique", "un graphique linéaire", "un graphique en barres",
        "a chart", "a line chart", "a bar chart", "a plot", "a time-series chart"
    ],
    "tableau": [
        "un tableau de données", "tableau en grille", "tableau financier",
        "a data table", "a grid table", "a spreadsheet-like table", "tabular data"
    ],
    "schéma": [
        "un schéma", "un diagramme", "schéma d’architecture",
        "a schematic diagram", "a flowchart", "a block diagram", "a network diagram"
    ],
    "logo": [
        "un logo", "logo d’entreprise", "emblème",
        "a corporate logo", "a brand logo", "an emblem", "a company logo"
    ],
    "autre": [
        "photo générale", "texte simple", "illustration diverse",
        "generic photo", "plain text page", "miscellaneous illustration", "other content"
    ],
}

class ImageZeroShotClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score_image_vs_prompts(
        self, image: Image.Image, prompts: List[str]
    ) -> torch.Tensor:
        """
        Retourne un vecteur de probabilités (softmax) de taille len(prompts).
        On encode l'image une fois, et tous les prompts en batch.
        """
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        # Similarités image<->text; CLIP sort déjà dans un espace commun
        logits_per_image = outputs.logits_per_image.squeeze(0)  # [num_prompts]
        probs = logits_per_image.softmax(dim=-1)  # softmax across prompts
        return probs.detach().cpu()

    def classify_image(
        self,
        image_path: str,
        aggregation: str = "max"  # "max" ou "mean"
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classe une image parmi les classes définies dans CLASS_PROMPTS.
        Agrégation des prompts d'une classe par max (par défaut) ou mean.
        Retour:
          - best_class : classe prédite
          - class_scores : dict {classe -> score agrégé}
        """
        img = Image.open(image_path).convert("RGB")

        class_scores: Dict[str, float] = {}
        for cls, prompts in CLASS_PROMPTS.items():
            probs = self.score_image_vs_prompts(img, prompts)  # [len(prompts)]
            if aggregation == "mean":
                score = float(probs.mean())
            else:  # "max"
                score = float(probs.max())
            class_scores[cls] = score

        best_class = max(class_scores, key=class_scores.get)
        return best_class, class_scores

    def classify_folder(
        self,
        folder: str,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
        aggregation: str = "max"
    ) -> Dict[str, Dict[str, float]]:
        """
        Classe toutes les images d'un dossier.
        Retour: {image_path: {"pred": classe, "scores": {classe: score, ...}}}
        """
        results = {}
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path) or not name.lower().endswith(extensions):
                continue
            pred, scores = self.classify_image(path, aggregation=aggregation)
            results[path] = {"pred": pred, "scores": scores}
        return results

def pretty(scores: Dict[str, float]) -> str:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}: {v:.3f}" for k, v in items])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Chemin d'une image à tester")
    parser.add_argument("--folder", type=str, help="Dossier d'images à classifier")
    parser.add_argument("--agg", type=str, default="max", choices=["max", "mean"], help="Agrégation des prompts par classe")
    args = parser.parse_args()

    clf = ImageZeroShotClassifier()

    if args.image:
        pred, scores = clf.classify_image(args.image, aggregation=args.agg)
        print(f"[IMAGE] {args.image}\n→ Classe: {pred}\n→ Scores: {pretty(scores)}")
    elif args.folder:
        res = clf.classify_folder(args.folder, aggregation=args.agg)
        for p, meta in res.items():
            print(f"[FOLDER] {p}\n→ Classe: {meta['pred']}\n→ Scores: {pretty(meta['scores'])}")
    else:
        print("Spécifie --image <path> ou --folder <dir>")
