# pipeline/cache_manager.py
import hashlib
import json
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Optional

class EmbeddingCacheManager:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _pdf_hash(self, file_path: Path) -> str:
        """Calcule un hash unique pour le fichier."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def get_cache_paths(self, file_path: Path) -> Tuple[Path, Path, Path]:
        """Retourne les chemins (chunks, embeddings, index) pour ce fichier."""
        file_hash = self._pdf_hash(file_path)
        cache_path = self.cache_dir / file_hash
        cache_path.mkdir(parents=True, exist_ok=True)
        return (
            cache_path / "chunks.json",
            cache_path / "embeddings.npy",
            cache_path / "index.pkl"
        )

    def load_cache(self, file_path: Path) -> Optional[Tuple[List[str], np.ndarray, object]]:
        """Charge le cache si disponible."""
        chunks_file, emb_file, index_file = self.get_cache_paths(file_path)
        if chunks_file.exists() and emb_file.exists() and index_file.exists():
            chunks = json.load(open(chunks_file, "r", encoding="utf-8"))
            embeddings = np.load(emb_file)
            index = joblib.load(index_file)
            return chunks, embeddings, index
        return None

    def save_cache(self, file_path: Path, chunks: List[str], embeddings: np.ndarray, index: object):
        """Sauvegarde les données dans le cache."""
        chunks_file, emb_file, index_file = self.get_cache_paths(file_path)
        json.dump(chunks, open(chunks_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        np.save(emb_file, embeddings)
        joblib.dump(index, index_file)
