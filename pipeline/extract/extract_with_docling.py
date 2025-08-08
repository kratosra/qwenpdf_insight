# pdf_docling_pipeline.py
# ───────────────────────
"""
Étape 1 : PDF → Markdown (via Docling)
Étape 2 : découpe du Markdown en *chunks* de longueur ≈ chunk_size
          • un tableau Markdown (| … |) reste toujours dans le même chunk
"""

from __future__ import annotations

import json, logging, re, textwrap
from pathlib import Path
from typing     import List

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models       import InputFormat
from docling.document_converter          import DocumentConverter, PdfFormatOption


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# 1)  PDF  ➜  MARKDOWN
# ────────────────────────────────────────────────────────────────
def pdf_to_markdown(pdf_path: str | Path,
                    output_dir: str | Path = "docling_output") -> Path:
    """Convertit un PDF en Markdown + exports JSON/TXT/DOCTAGS.

    • Retourne le chemin du .md généré
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    out_dir  = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )
    # meilleur appariement cellule/label si possible
    opts.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )

    log.info("→ Docling conversion … (%s)", pdf_path.name)
    result   = converter.convert(str(pdf_path))
    doc      = result.document
    stem     = pdf_path.stem

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    log.info("   Markdown exporté  →  %s", md_path.relative_to(Path.cwd()))

    # (optionnel) autres formats pour debug / inspection
    (out_dir / f"{stem}.json"   ).write_text(json.dumps(doc.export_to_dict(),
                                                        ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (out_dir / f"{stem}.txt"    ).write_text(doc.export_to_text(),    encoding="utf-8")
    (out_dir / f"{stem}.doctags").write_text(doc.export_to_doctags(), encoding="utf-8")
    return md_path


# ────────────────────────────────────────────────────────────────
# 2)  MARKDOWN  ➜  CHUNKS
# ────────────────────────────────────────────────────────────────
# Un tableau est détecté dès qu’une ligne contient un “|” encadré par au
# moins un caractère de texte ; le bloc est ensuite agrégé jusqu’à la
# prochaine ligne vide qui n’appartient pas au tableau.
TABLE_SEP_RE = re.compile(r"\|.*\|")

def _collect_table(lines_iter) -> str:
    """Récupère un bloc-table Markdown complet à partir de la première ligne '| … |'."""
    collected = []
    for line in lines_iter:
        if not line.strip():      # ligne vide = fin de tableau
            collected.append(line)
            break
        collected.append(line)
    return "".join(collected)


def chunk_markdown(md_path: str | Path,
                   chunk_size: int = 2048,
                   output_chunks_path: str | Path | None = None,
                   sliding_overlap: int = 200
                   ) -> List[str]:
    """Découpe le markdown en *chunks* ≤ chunk_size,
       en préservant chaque tableau ou description d'image Markdown entier,
       avec une fenêtre glissante entre les chunks de texte.
    """
    md_path = Path(md_path).expanduser().resolve()
    chunks  : list[str] = []
    current : list[str] = []
    cur_len = 0

    with md_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        if TABLE_SEP_RE.search(line):
            table_block = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                table_block.append(lines[i])
                i += 1
            if i < len(lines):
                table_block.append(lines[i])
            table_text = "".join(table_block).rstrip()

            if cur_len + len(table_text) > chunk_size and cur_len > 0:
                chunks.append("".join(current).strip())
                current, cur_len = [], 0
            current.append(table_text + "\n\n")
            cur_len += len(table_text)
            i += 1
            continue

        if cur_len + len(line) > chunk_size:
            chunks.append("".join(current).strip())
            # sliding window: keep last N characters
            sliding_text = "".join(current)[-sliding_overlap:]
            current = [sliding_text]
            cur_len = len(sliding_text)
        current.append(line)
        cur_len += len(line)
        i += 1

    if current:
        chunks.append("".join(current).strip())

    if output_chunks_path:
        output_chunks_path = Path(output_chunks_path).expanduser().resolve()
        output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
        output_chunks_path.write_text("\n---\n".join(chunks), encoding="utf-8")
        log.info("   Fichier chunks → %s", output_chunks_path.relative_to(Path.cwd()))

    log.info("   %d chunks générés (≤ %d car.)", len(chunks), chunk_size)
    return chunks

"""
# ────────────────────────────────────────────────────────────────
# Exemple
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PDF_FILE    = "graph_discovery/project_qwen/DocFinancier.pdf"
    EXPORT_DIR  = "graph_discovery/project_qwen/docling_md"
    CHUNK_FILE  = "graph_discovery/project_qwen/docling_chunks/Docling_chunks.txt"

    # 1) PDF → MD
    md_file = pdf_to_markdown(PDF_FILE, EXPORT_DIR)

    # 2) MD → chunks (tableaux conservés tels quels)
    chunks  = chunk_markdown(md_file,
                             chunk_size=600,
                             output_chunks_path=CHUNK_FILE)

    print(f"\n {len(chunks)} chunks prêts pour l’embedding.")
"""