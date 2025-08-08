import re
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _is_table_like(text: str,
                   digit_ratio: float = 0.30,
                   letter_ratio: float = 0.60,
                   min_numbers: int = 12) -> bool:
    """Return **True** iff the chunk is overwhelmingly numeric.

    Conditions (all must be met):
    1. digits/commas/dots/% ≥ ``digit_ratio`` of total chars.
    2. alphabetic chars ≤ ``letter_ratio``.
    3. at least ``min_numbers`` numeric tokens.
    """
    text = text.strip()
    if not text:
        return False

    total = len(text)
    digits = sum(c.isdigit() or c in {",", ".", "%"} for c in text)
    letters = sum(c.isalpha() for c in text)
    n_nums = len(re.findall(r"\b\d+[\d.,]*%?\b", text))

    return (digits / total) >= digit_ratio and (letters / total) <= letter_ratio and n_nums >= min_numbers


def _clean_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip()) if cell else ""


def _merge_header_rows(rows):
    max_cols = max(len(r) for r in rows)
    merged = []
    for c in range(max_cols):
        pieces = [rows[r][c] for r in range(len(rows)) if c < len(rows[r]) and rows[r][c]]
        merged.append(_clean_cell(" ".join(pieces)))
    return merged


def _markdown_table(raw_rows):
    header_rows, data_rows = [], []
    for row in raw_rows:
        numeric_cells = sum(bool(re.search(r"\d", cell or "")) for cell in row)
        if data_rows or numeric_cells >= 2:
            data_rows.append(row)
        else:
            header_rows.append(row)
    if not header_rows:
        header_rows = [raw_rows[0]]
        data_rows = raw_rows[1:]

    headers = _merge_header_rows(header_rows)
    headers = [h if h else f"col{i+1}" for i, h in enumerate(headers)]

    md = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for r in data_rows:
        cells = [_clean_cell(r[i]) if i < len(r) else "" for i in range(len(headers))]
        md.append(" | ".join(cells))
    return "\n".join(md)

# ------------------------------------------------------------
# Main extractor
# ------------------------------------------------------------

def extract_all_pdf_content(pdf_path: str,
                            output_log_txt: str = "pdf_extraction_log.txt",
                            chunk_size: int = 600):
    text_chunks, image_entries = [], []

    with open(output_log_txt, "w", encoding="utf-8") as log:
        log.write(f"PDF extraction log → {pdf_path}\n\n")

        with pdfplumber.open(pdf_path) as pdf:
            for p_num, page in enumerate(pdf.pages, 1):
                log.write(f"\n=== Page {p_num} ===\n")

                # Narrative text split + filter numeric-only blocks
                raw_txt = page.extract_text() or ""
                if raw_txt.strip():
                    parts = [raw_txt[i:i + chunk_size] for i in range(0, len(raw_txt), chunk_size)]
                    for idx, part in enumerate(parts, 1):
                        if _is_table_like(part):
                            continue  # skip noisy numeric dump
                        log.write(f"\n[Text {idx}]\n{part.strip()}\n")
                        text_chunks.append(part.strip())

                # Tables → Markdown
                for t_idx, tbl in enumerate(page.extract_tables() or [], 1):
                    if not tbl or not any(tbl):
                        continue
                    md = _markdown_table([[c or "" for c in row] for row in tbl])
                    md_block = f"Table p{p_num}-{t_idx}:\n" + md
                    log.write("\n" + md_block + "\n")
                    text_chunks.append(md_block)

        # Images
        log.write("\n=== IMAGES ===\n")
        doc = fitz.open(pdf_path)
        for p_idx in range(len(doc)):
            for img_idx, img in enumerate(doc[p_idx].get_images(full=True), 1):
                pil = Image.open(BytesIO(doc.extract_image(img[0])["image"])).convert("RGB")
                image_entries.append({"page": p_idx + 1, "index": img_idx, "image": pil})
                log.write(f"Image p{p_idx + 1}-{img_idx} size={pil.size}\n")

    print(f" Extraction complete → {output_log_txt}")
    return text_chunks, image_entries

"""
chunks, images = extract_all_pdf_content("graph_discovery/project_qwen/jepa.pdf", "graph_discovery/log_my_doc.txt")
print(f"{len(chunks)} chunks ready for embedding.")
print(f"{len(images)} images ready for visual Q&A.")
"""