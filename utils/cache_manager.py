# pipeline/utils/cache_manager.py

import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path, chunk_size=8192) -> str:
    """Calcule le hash SHA256 d'un fichier pour l'identifier de façon unique."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def get_cached_paths(hash_id: str, base_dir: Path) -> dict:
    """Retourne les chemins où les fichiers devraient être sauvegardés en cache."""
    return {
        "pdf": base_dir / f"{hash_id}.pdf",
        "markdown": base_dir / "markdown" / f"{hash_id}.md",
        "chunks": base_dir / "chunks" / f"{hash_id}_chunks.txt"
    }

def is_cached(paths: dict) -> bool:
    """Vérifie si tous les fichiers nécessaires sont déjà en cache."""
    return all(p.exists() for p in paths.values())
