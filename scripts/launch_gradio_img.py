# scripts/launch_app.py

import sys
from pathlib import Path

# Ajoute le dossier racine (qwenpdf_insight) au PYTHONPATH dynamiquement
sys.path.append(str(Path(__file__).resolve().parents[1]))


from app.qwen_interface_docling_img import demo

if __name__ == "__main__":
    demo.launch()
