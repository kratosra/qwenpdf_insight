# pdf_docling_pipeline.py
# ───────────────────────────────────────────
"""
Étape 1 : PDF → Markdown (via Docling avec image captioning)
Étape 2 : découpe du Markdown en *chunks* de longueur ≈ chunk_size
          • un tableau Markdown (| … |) reste toujours dans le même chunk
          • une description d'image (commence par "![") est conservée en entier
"""

from __future__ import annotations

import json, logging, re
from pathlib import Path
from typing import List

from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 1)  PDF  ➜  MARKDOWN avec VLM pour images
# ─────────────────────────────────────────

def pdf_to_markdown(pdf_path: str | Path,
                    output_dir: str | Path = "docling_output") -> Path:
    """Convertit un PDF en Markdown enrichi avec descriptions d'images,
    tableaux, et texte OCR grâce à Docling + Qwen-VL.

    • Retourne le chemin du fichier .md généré
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    out_dir  = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = pdf_path.stem

    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_picture_description=True,
        generate_picture_images=True,
        images_scale=2.0,
    )

    opts.picture_description_options = PictureDescriptionVlmOptions(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt=(
            "If the image contains a table, extract the data and present it in a clean, "
            "well-formatted Markdown table. Otherwise, provide a detailed and accurate "
            "description of the image content."
        )
    )

    opts.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )

    log.info("→ Conversion Docling avec image captioning (%s)", pdf_path.name)
    result = converter.convert(str(pdf_path))
    doc    = result.document

    # Export Markdown
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    log.info("   Markdown exporté  →  %s", md_path.relative_to(Path.cwd()))

    # Exports supplémentaires (optionnels)
    (out_dir / f"{stem}.json"   ).write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{stem}.txt"    ).write_text(doc.export_to_text(),    encoding="utf-8")
    (out_dir / f"{stem}.doctags").write_text(doc.export_to_doctags(), encoding="utf-8")

    return md_path

# ─────────────────────────────────────────
# 2)  MARKDOWN  ➜  CHUNKS
# ─────────────────────────────────────────

TABLE_SEP_RE = re.compile(r"\|.*\|")
IMG_DESC_RE = re.compile(r"^!\[.*\]\(.*\)$")

def _collect_table(lines_iter) -> str:
    """Récupère un bloc-table Markdown complet à partir de la première ligne '| … |'."""
    collected = []
    for line in lines_iter:
        if not line.strip():
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

        if IMG_DESC_RE.match(line):
            if cur_len > 0:
                chunks.append("".join(current).strip())
                current, cur_len = [], 0
            current.append(line)
            chunks.append("".join(current).strip())
            current, cur_len = [], 0
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
