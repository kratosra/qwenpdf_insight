# evaluate_vlm_vs_pdf.py
from pathlib import Path
import json
import re
import csv
import fitz
from typing import List, Dict, Tuple

def norm(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w,\.\-\+%/ ]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def lcs(a: List[str], b: List[str]) -> int:
    # ROUGE-L (LCS length)
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j]+1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def f1_tokens(ref_toks: List[str], hyp_toks: List[str]) -> Tuple[float,float,float]:
    ref_set = set(ref_toks); hyp_set = set(hyp_toks)
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    f1 = 2*p*r / max(1e-9, (p+r))
    return p, r, f1

_num_re = re.compile(r"[-+]?\d+(?:[\.,]\d+)?%?")
def extract_numbers(text: str) -> List[str]:
    # garde forme textuelle, normalise la virgule en point pour comparaison
    nums = _num_re.findall(text)
    return [n.replace(",", ".") for n in nums]

def num_compare(ref_nums: List[str], hyp_nums: List[str], tol: float = 0.01) -> Dict[str, float]:
    # calcule proportion des nombres hyp qui matchent un nombre ref +/- tol relative (ou égalité string pour %)
    ref_parsed = []
    for x in ref_nums:
        if x.endswith("%"):
            try:
                ref_parsed.append(("pct", float(x[:-1])))
            except:
                ref_parsed.append(("raw", x))
        else:
            try:
                ref_parsed.append(("num", float(x)))
            except:
                ref_parsed.append(("raw", x))

    hyp_parsed = []
    for x in hyp_nums:
        if x.endswith("%"):
            try:
                hyp_parsed.append(("pct", float(x[:-1])))
            except:
                hyp_parsed.append(("raw", x))
        else:
            try:
                hyp_parsed.append(("num", float(x)))
            except:
                hyp_parsed.append(("raw", x))

    matched = 0
    for t, v in hyp_parsed:
        found = False
        for t2, u in ref_parsed:
            if t == "raw" or t2 == "raw":
                if str(v) == str(u):
                    found = True; break
            elif t == t2 == "pct":
                if abs(v - u) <= max(0.1, tol*max(abs(u),1.0)):  # 0.1 pt d'écart min
                    found = True; break
            elif t == t2 == "num":
                if abs(v - u) <= tol*max(abs(u),1.0):
                    found = True; break
        if found:
            matched += 1

    hyp_total = max(1, len(hyp_parsed))
    ref_total = max(1, len(ref_parsed))
    return {
        "hyp_in_ref_ratio": matched / hyp_total,     # % des nombres du VLM retrouvés dans le PDF (moins d’hallucinations)
        "ref_covered_ratio": matched / ref_total,    # % des nombres du PDF repris par le VLM (couverture)
        "hyp_count": len(hyp_parsed),
        "ref_count": len(ref_parsed),
    }

def extract_ref_text_from_bbox(pdf_path: Path, page_1based: int, bbox_72dpi: List[float]) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_1based - 1]
    rect = fitz.Rect(*bbox_72dpi)
    # PyMuPDF extrait le texte des objets vectoriels dans cette zone
    text = page.get_text("text", clip=rect)
    doc.close()
    return text or ""

def main(
    pdf_path: str,
    results_json: str,     # ton results.json (sortie du pipeline)
    figures_index_json: str # figures/figures_index.json (métadonnées de crops)
):
    pdf_path = Path(pdf_path)
    out = json.loads(Path(results_json).read_text(encoding="utf-8"))
    index = json.loads(Path(figures_index_json).read_text(encoding="utf-8"))

    # Map image -> bbox info
    meta_by_png = { entry["png_path"]: entry for entry in index }

    rows = []
    for item in out.get("per_image", []):
        png = item["image"]
        ans = item.get("answer","")
        meta = meta_by_png.get(png)
        if not meta:
            rows.append({
                "image": png,
                "status": "NO_META",
                "p": 0, "r": 0, "f1": 0,
                "rougeL": 0,
                "num_hyp_in_ref": 0, "num_ref_covered": 0,
                "ref_len": 0, "hyp_len": len(ans),
                "ref_text": "", "hyp_text": ans,
            })
            continue

        ref_text = extract_ref_text_from_bbox(pdf_path, meta["page_index_1based"], meta["bbox_72dpi"])

        # tokens
        ref_toks = norm(ref_text)
        hyp_toks = norm(ans)

        # scores textuels
        p, r, f1 = f1_tokens(ref_toks, hyp_toks)
        lcs_len = lcs(ref_toks, hyp_toks)
        rougeL = 2*lcs_len / max(1, len(ref_toks)+len(hyp_toks))

        # nombres
        nums_ref = extract_numbers(ref_text)
        nums_hyp = extract_numbers(ans)
        num_stats = num_compare(nums_ref, nums_hyp, tol=0.01)

        rows.append({
            "image": png,
            "status": "OK",
            "p": round(p,4), "r": round(r,4), "f1": round(f1,4),
            "rougeL": round(rougeL,4),
            "num_hyp_in_ref": round(num_stats["hyp_in_ref_ratio"],4),
            "num_ref_covered": round(num_stats["ref_covered_ratio"],4),
            "ref_len": len(ref_toks), "hyp_len": len(hyp_toks),
            "ref_text": ref_text[:8000],  # garde raisonnable
            "hyp_text": ans[:8000],
        })

    # Sauvegardes
    # JSON détaillé
    Path("eval_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV synthétique
    with open("eval_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","status","f1","rougeL","p","r","num_hyp_in_ref","num_ref_covered","ref_len","hyp_len"])
        for r in rows:
            w.writerow([r["image"], r["status"], r["f1"], r["rougeL"], r["p"], r["r"], r["num_hyp_in_ref"], r["num_ref_covered"], r["ref_len"], r["hyp_len"]])

    # Rapport lisible
    with open("eval_report.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write("="*80 + "\n")
            f.write(f"Image: {r['image']} | Status: {r['status']}\n")
            f.write(f"F1={r['f1']}  ROUGE-L={r['rougeL']}  P={r['p']}  R={r['r']}\n")
            f.write(f"Num match (hyp∈ref)={r['num_hyp_in_ref']}  Couverture ref={r['num_ref_covered']}\n")
            f.write("- Référence (PDF zone):\n")
            f.write(r["ref_text"] + "\n")
            f.write("- Réponse VLM:\n")
            f.write(r["hyp_text"] + "\n\n")

    print("OK -> eval_report.json / eval_report.csv / eval_report.txt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Chemin vers le PDF original")
    ap.add_argument("--results", default="results.json", help="Sortie VLM (pipeline) JSON")
    ap.add_argument("--index", default="data/images/<HASH>/figures/figures_index.json", help="Index des figures")
    args = ap.parse_args()
    main(args.pdf, args.results, args.index)
