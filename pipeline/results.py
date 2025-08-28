#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys, time, json, csv, traceback, re
from datetime import datetime
import numpy as np

# =========================
# Configs par défaut
# =========================
DEFAULT_RESULTS_CSV = "data/reports/results.csv"
DEFAULT_REPORT_DIR  = "data/reports"
DEFAULT_TMP_DIR     = "data/tmp"
DEFAULT_EVAL_DIR    = "eval"
DEFAULT_RUNS_DIR    = "eval/runs"
DEFAULT_OUT_DIR     = "eval/out"

# =========================
# PYTHONPATH racine projet
# =========================
sys.path.append(str(Path(__file__).resolve().parents[1]))

# === Imports pipeline existants ===
from pipeline.extract.extract_with_docling import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import (
    embed_chunks_qwen3,
    build_faiss_index,
    retrieve_top_k_chunks,
)
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format


# =========================
# Utils / instrumentation
# =========================
def _now_ms() -> float:
    return time.perf_counter() * 1000.0

def _preview(text: str, n: int = 160) -> str:
    return text[:n].replace("\n", " ").strip() + ("…" if len(text) > n else "")

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------- Nettoyage des sorties LLM : suppression du raisonnement (<think>...</think>) ---------
_THINK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_OPEN_RE = re.compile(r"(?is)<think>.*")

def clean_answer(text: str) -> str:
    """
    Supprime les segments privés (<think>...</think>) et normalise légèrement les espaces.
    """
    if not isinstance(text, str):
        return ""
    t = _THINK_RE.sub("", text)
    t = _THINK_OPEN_RE.sub("", t)
    # lignes 'thoughts:' au début (par sécurité)
    t = re.sub(r"(?im)^\s*thoughts?\s*:\s*.*$", "", t)
    # normalise espaces
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def normalize_text(s: str) -> str:
    """
    Normalisation textuelle douce (minuscule, retire ponctuation forte sauf %).
    Conserve chiffres, lettres, %, espaces.
    """
    s = (s or "").lower()
    s = s.replace("\u202f", " ").replace("\xa0", " ")  # espaces fines/insécables -> espace
    s = re.sub(r"[^0-9a-zàâçéèêëîïôûùüÿñæœ%\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----- Extraction de nombres/percentages (EM numérique robuste) -----
_NUM_TOKEN = re.compile(
    r"(?ix)"                # ignore case, verbose
    r"(\+|-)?\s*"           # signe optionnel
    r"(?:\d{1,3}(?:[\s.,]\d{3})+|\d+)"  # parties milliers: 1 234 | 1,234 | 1234
    r"(?:[.,]\d+)?\s*"      # décimales optionnelles
    r"(?:%|mmad|mad|k|m|b)?"# unité simple optionnelle (élargissable)
)

def _canon_number(tok: str) -> str:
    """
    Canonicalise un token numérique :
    - retire espaces fines/insécables
    - unifie séparateurs de milliers
    - conserve % et quelques unités simples
    """
    t = tok.lower().replace("\u202f", " ").replace("\xa0", " ")
    t = t.strip()
    # sépare unité finale simple si présente
    m = re.match(r"^(.+?)(%|mmad|mad|k|m|b)?$", t)
    core = (m.group(1) if m else t).strip()
    unit = (m.group(2) if m else "").strip()

    # retire espaces dans la partie nombre, remplace virgule décimale par point
    core = core.replace(" ", "")
    # si virgule utilisée comme décimale (ex: 16,5) -> 16.5
    # mais si core contient à la fois , et . on remplace les , de milliers
    if core.count(",") == 1 and core.count(".") == 0:
        core = core.replace(",", ".")
    else:
        core = core.replace(",", "")
    # remove leading plus
    core = core.lstrip("+")
    return core + unit

def extract_numbers(s: str) -> List[str]:
    return [_canon_number(m.group(0)) for m in _NUM_TOKEN.finditer(s)]

def numbers_em(pred: str, gold: str) -> int:
    """
    EM numérique : les multisets de nombres/percentages doivent correspondre (ordre indifférent).
    Utile pour des réponses financières (ex: '4 830 MMAD', '90%').
    """
    p = sorted(extract_numbers(normalize_text(pred)))
    g = sorted(extract_numbers(normalize_text(gold)))
    return int(p == g)

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def f1_span(pred: str, gold: str) -> float:
    from collections import Counter
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(p) if p else 0.0
    recall    = num_same / len(g) if g else 0.0
    return 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

def rouge_l(pred: str, gold: str) -> float:
    a = normalize_text(pred).split()
    b = normalize_text(gold).split()
    if not a or not b: return 0.0
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1]==b[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    prec = lcs/len(a)
    rec  = lcs/len(b)
    return (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0


class StatsCollector:
    def __init__(self, report_dir: Path, run_id: str):
        self.timing: Dict[str, float] = {}
        self.meta: Dict[str, Any] = {}
        self.retrieval: Dict[str, Any] = {}
        self.generation: Dict[str, Any] = {}
        self.report_dir = _ensure_dir(report_dir)
        self.run_id = run_id

    def set_timing(self, **kwargs: float) -> None: self.timing.update(kwargs)
    def set_meta(self, **kwargs: Any) -> None: self.meta.update(kwargs)
    def set_retrieval(self, **kwargs: Any) -> None: self.retrieval.update(kwargs)
    def set_generation(self, **kwargs: Any) -> None: self.generation.update(kwargs)

    def write_reports(self, prefix: str = "run") -> Tuple[Path, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{self.run_id}_{ts}"
        json_path = self.report_dir / f"{base}.json"
        md_path = self.report_dir / f"{base}.md"

        data = {
            "meta": self.meta,
            "timing_ms": self.timing,
            "retrieval": self.retrieval,
            "generation": self.generation,
        }
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = []
        lines.append(f"# Rapport d'exécution — {ts}\n")
        lines.append("## Métadonnées")
        for k, v in self.meta.items(): lines.append(f"- **{k}**: {v}")
        lines.append("\n## Temps par étape (ms)")
        for k, v in self.timing.items(): lines.append(f"- **{k}**: {v:.2f} ms")
        if self.retrieval:
            lines.append("\n## Retrieval")
            for k, v in self.retrieval.items():
                if k != "topk_details": lines.append(f"- **{k}**: {v}")
            details = self.retrieval.get("topk_details", [])
            if details:
                lines += ["", "### Détails Top-k", "| Rang | Similarité cos | Aperçu chunk |", "|---:|---:|---|"]
                for d in details:
                    lines.append(f"| {d['rank']} | {d['cosine_sim']:.4f} | {d['preview']} |")
        if self.generation:
            lines += ["", "## Génération"]
            for k, v in self.generation.items():
                if k != "answer_preview": lines.append(f"- **{k}**: {v}")
            if "answer_preview" in self.generation:
                lines += ["", "**Aperçu de la réponse :**", "", self.generation["answer_preview"]]

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, md_path


# =========================
# Pipeline
# =========================
class QwenPDFPipeline:
    def __init__(
        self,
        chunk_size: int = 2048,
        default_top_k: int = 3,
        temp_dir: str = DEFAULT_TMP_DIR,
        report_dir: str = DEFAULT_REPORT_DIR,
        csv_path: str = DEFAULT_RESULTS_CSV,
        runs_dir: str = DEFAULT_RUNS_DIR,
        verbose: bool = True,
    ):
        self.chunk_size = chunk_size
        self.default_top_k = default_top_k
        self.temp_dir = _ensure_dir(Path(temp_dir))
        self.report_dir = Path(report_dir)
        self.csv_path = Path(csv_path)
        self.runs_dir = _ensure_dir(Path(runs_dir))
        self.verbose = verbose

    def log(self, msg: str):
        if self.verbose: print(f"[QwenPDFPipeline] {msg}")

    # --- Étapes unitaires ---
    def _extract_chunks(self, pdf_path: Path, stats: StatsCollector) -> List[str]:
        t0 = _now_ms()
        self.log("Extraction Markdown…")
        md_path = pdf_to_markdown(pdf_path, output_dir=self.temp_dir / "markdown")
        t1 = _now_ms()

        self.log("Découpage en chunks…")
        chunks = chunk_markdown(md_path, chunk_size=self.chunk_size, output_chunks_path=self.temp_dir / "chunks.txt")
        t2 = _now_ms()

        stats.set_timing(extraction_ms=t1 - t0, chunking_ms=t2 - t1)

        lengths = [len(c) for c in chunks]
        stats.set_meta(
            n_chunks=len(chunks),
            chunk_len_avg=float(np.mean(lengths)) if lengths else 0.0,
            chunk_len_min=int(np.min(lengths)) if lengths else 0,
            chunk_len_max=int(np.max(lengths)) if lengths else 0,
            chunk_size_param=self.chunk_size,
        )
        return chunks

    def _retrieve(self, chunks: List[str], question: str, top_k: int, stats: StatsCollector) -> Tuple[List[str], Dict[str, Any]]:
        t0 = _now_ms()
        self.log("Embedding des chunks…")
        embeddings = embed_chunks_qwen3(chunks)  # (N, d)
        t1 = _now_ms()

        self.log("Indexation FAISS…")
        index = build_faiss_index(embeddings)
        t2 = _now_ms()

        self.log(f"Recherche Top-{top_k}…")
        top_chunks = retrieve_top_k_chunks(question, chunks, embeddings, index, top_k=top_k)
        t3 = _now_ms()

        stats.set_timing(embedding_ms=t1 - t0, index_ms=t2 - t1, search_ms=t3 - t2)

        # Similarités cos pour diagnostic
        q_vec = embed_chunks_qwen3([question])[0]
        emb_norms = np.linalg.norm(embeddings, axis=1)
        qn = np.linalg.norm(q_vec) or 1.0
        sims = (embeddings @ q_vec) / (emb_norms * qn + 1e-12)

        text_to_first_idx: Dict[str, int] = {}
        for i, c in enumerate(chunks):
            if c not in text_to_first_idx: text_to_first_idx[c] = i

        order = np.argsort(-sims)
        rank_of_idx = {int(idx): int(rk) for rk, idx in enumerate(order)}

        topk_details = []
        topk_indices = []
        for c in top_chunks:
            idx = text_to_first_idx.get(c, None)
            if idx is None: continue
            topk_indices.append(int(idx))
            topk_details.append({
                "rank": rank_of_idx[idx] + 1,
                "index": idx,
                "cosine_sim": float(sims[idx]),
                "preview": _preview(c),
            })

        stats.set_retrieval(
            top_k=top_k,
            sim_mean=float(np.mean(sims)),
            sim_std=float(np.std(sims)),
            sim_p90=float(np.percentile(sims, 90)),
            sim_best=float(np.max(sims)),
            topk_details=sorted(topk_details, key=lambda d: d["rank"]),
        )
        ctx = {"embeddings": embeddings, "index": index, "sims": sims,
               "topk_indices": topk_indices, "topk_texts": top_chunks}
        return top_chunks, ctx

    def _generate(self, top_chunks: List[str], question: str, stats: StatsCollector) -> str:
        t0 = _now_ms()
        self.log("Génération de la réponse…")
        raw_answer = generate_answer_qwen_chat_format(top_chunks, question)
        answer = clean_answer(raw_answer)  # <-- nettoyage automatique
        t1 = _now_ms()
        stats.set_timing(generation_ms=t1 - t0)
        stats.set_generation(answer_len=len(answer), answer_preview=_preview(answer, 320))
        return answer  # <-- on renvoie la réponse complète nettoyée

    # --- Routing des types de questions → top_k ---
    def _route(self, q_type: str) -> Dict[str, Any]:
        q_type = (q_type or "Autre").strip().lower()
        if q_type in ("résumé", "resume"):
            return {"top_k": max(5, self.default_top_k)}
        if q_type in ("valeur", "valeurs"):
            return {"top_k": 3}
        if q_type == "image":
            return {"top_k": 4}
        return {"top_k": self.default_top_k}

    # ------------------------- Exécutions -------------------------
    def run_single(self, pdf_path: str, question: str, q_type: str, run_id: str, qid: str = "") -> Dict[str, Any]:
        stats = StatsCollector(report_dir=self.report_dir, run_id=run_id)
        rec: Dict[str, Any] = {}
        try:
            pdfp = Path(pdf_path)
            route = self._route(q_type)
            top_k = int(route["top_k"])
            stats.set_meta(pdf=str(pdfp), question=question, question_type=q_type, top_k_used=top_k, run_id=run_id)

            chunks = self._extract_chunks(pdfp, stats)
            if not chunks:
                stats.set_meta(status="empty_chunks")
                json_p, md_p = stats.write_reports(prefix="run")
                rec = self._row_from_stats(stats, answer="", status="empty_chunks", json_p=json_p, md_p=md_p)
                return rec

            top_chunks, ctx = self._retrieve(chunks, question, top_k, stats)
            answer = self._generate(top_chunks, question, stats)

            stats.set_meta(status="ok")
            json_p, md_p = stats.write_reports(prefix="run")
            rec = self._row_from_stats(stats, answer=answer, status="ok", json_p=json_p, md_p=md_p)

            # --- JSONL: log complet pour retrieval metrics & analyse ---
            run_jsonl = self.runs_dir / f"{run_id}.jsonl"
            with run_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "qid": qid or run_id,
                    "pdf": str(pdfp),
                    "question": question,
                    "question_type": q_type,
                    "topk_indices": ctx.get("topk_indices", []),
                    "topk_texts": ctx.get("topk_texts", []),
                    "answer_pred": answer  # <-- full answer nettoyée
                }, ensure_ascii=False) + "\n")
            rec["run_jsonl"] = str(run_jsonl)

            return rec

        except Exception as e:
            err = f"Erreur : {e}\n{traceback.format_exc()}"
            self.log(err)
            stats.set_meta(status="error", error=str(e))
            json_p, md_p = stats.write_reports(prefix="error")
            rec = self._row_from_stats(stats, answer="", status="error", json_p=json_p, md_p=md_p)
            return rec

    def run_batch(self, pdf_path: str, questions: List[Dict[str, str]], batch_id: str = "", gold_path: str | None = None) -> List[Dict[str, Any]]:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id_base = batch_id or run_stamp
        rows: List[Dict[str, Any]] = []
        for i, q in enumerate(questions, 1):
            q_type = q.get("type", "Autre")
            q_text = q.get("text", "")
            qid    = q.get("qid", f"{run_id_base}_q{i}")
            rec = self.run_single(pdf_path, q_text, q_type, run_id=f"{run_id_base}_q{i}", qid=qid)
            rows.append(rec)

        # écriture/append CSV agrégé (robuste aux clés en plus)
        self._append_csv(rows)

        # si gold fourni, lancer l'éval réponses automatiquement + AFFICHER SCORES
        if gold_path:
            try:
                perq_csv, summary_json, report_md = evaluate_answers(
                    results_csv=self.csv_path,
                    qa_gold_path=Path(gold_path),
                    out_dir=Path(DEFAULT_OUT_DIR)
                )
                # Lecture des scores pour affichage console
                summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
                print("\n===== SCORES RAG (réponses) =====")
                print(f"- N questions évaluées : {summary.get('n_questions')}")
                print(f"- Exact Match strict   : {summary.get('avg_EM_strict')}")
                print(f"- Exact Match numérique: {summary.get('avg_EM_numbers')}")
                print(f"- F1 token (moy.)      : {summary.get('avg_F1')}")
                print(f"- ROUGE-L (moy.)       : {summary.get('avg_ROUGE-L')}")
                print("=================================\n")
                print(f"[ÉVAL] Détails par question  → {perq_csv}")
                print(f"[ÉVAL] Résumé des scores     → {summary_json}")
                print(f"[ÉVAL] Rapport Markdown      → {report_md}")
            except Exception as e:
                print(f"[WARN] Échec évaluation réponses: {e}")

        return rows

    # ------------------------- CSV helpers -------------------------
    def _row_from_stats(self, stats: StatsCollector, answer: str, status: str, json_p: Path, md_p: Path) -> Dict[str, Any]:
        timing = stats.timing
        meta = stats.meta
        retr = stats.retrieval
        gen = stats.generation

        first_rank = None
        if retr.get("topk_details"):
            first_rank = retr["topk_details"][0]["rank"]

        return {
            "run_id": meta.get("run_id", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pdf": meta.get("pdf", ""),
            "question_type": meta.get("question_type", ""),
            "question": meta.get("question", ""),
            "status": status,
            "top_k": meta.get("top_k_used", ""),
            "n_chunks": meta.get("n_chunks", 0),
            "chunk_len_avg": meta.get("chunk_len_avg", 0.0),
            "chunk_len_min": meta.get("chunk_len_min", 0),
            "chunk_len_max": meta.get("chunk_len_max", 0),
            "extraction_ms": round(timing.get("extraction_ms", 0.0), 2),
            "chunking_ms": round(timing.get("chunking_ms", 0.0), 2),
            "embedding_ms": round(timing.get("embedding_ms", 0.0), 2),
            "index_ms": round(timing.get("index_ms", 0.0), 2),
            "search_ms": round(timing.get("search_ms", 0.0), 2),
            "generation_ms": round(timing.get("generation_ms", 0.0), 2),
            "sim_mean": round(retr.get("sim_mean", 0.0), 6),
            "sim_std": round(retr.get("sim_std", 0.0), 6),
            "sim_p90": round(retr.get("sim_p90", 0.0), 6),
            "sim_best": round(retr.get("sim_best", 0.0), 6),
            "best_rank": first_rank if first_rank is not None else "",
            "answer_len": gen.get("answer_len", 0),
            "answer_preview": gen.get("answer_preview", ""),
            "answer_full": answer,  # <-- réponse complète nettoyée (pour l'évaluation fidèle)
            "json_report": str(json_p),
            "md_report": str(md_p),
        }

    def _append_csv(self, rows: List[Dict[str, Any]]) -> None:
        _ensure_dir(self.csv_path.parent)
        fieldnames = [
            "run_id","timestamp","pdf","question_type","question","status",
            "top_k","n_chunks","chunk_len_avg","chunk_len_min","chunk_len_max",
            "extraction_ms","chunking_ms","embedding_ms","index_ms","search_ms","generation_ms",
            "sim_mean","sim_std","sim_p90","sim_best","best_rank",
            "answer_len","answer_preview","answer_full","json_report","md_report",
        ]
        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            # ignorer les champs non déclarés (ex: run_jsonl)
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for r in rows:
                safe_row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(safe_row)


# =========================
# Évaluation réponses (EM/F1/ROUGE-L) – fidèle
# =========================
def evaluate_answers(results_csv: Path, qa_gold_path: Path, out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # gold: jsonl avec {qid, question, answer_gold, ...}
    gold_map: Dict[str, Dict[str, str]] = {}
    with qa_gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            item = json.loads(line)
            gold_map[item["question"]] = item

    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv introuvable: {results_csv}")

    rows = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get("question","")
            if q in gold_map:
                rows.append(r)

    if not rows:
        raise RuntimeError("Aucune question du gold n'a été retrouvée dans results.csv. Exécute d'abord le batch fourni.")

    perq = []
    for r in rows:
        q = r.get("question","")
        # on évalue sur la réponse COMPLÈTE nettoyée (answer_full), fallback preview
        pred_full = r.get("answer_full","") or r.get("answer_preview","") or ""
        pred = clean_answer(pred_full)
        gold_ans = gold_map[q]["answer_gold"]

        em_strict   = exact_match(pred, gold_ans)
        em_numbers  = numbers_em(pred, gold_ans)
        f1          = f1_span(pred, gold_ans)
        rL          = rouge_l(pred, gold_ans)

        perq.append({
            "qid": gold_map[q]["qid"],
            "question": q,
            "answer_gold": gold_ans,
            "answer_pred": pred,
            "EM_strict": em_strict,
            "EM_numbers": em_numbers,
            "F1": round(f1,4),
            "ROUGE-L": round(rL,4),
        })

    perq_csv = Path(DEFAULT_OUT_DIR) / "answers_eval.csv"
    perq_csv.parent.mkdir(parents=True, exist_ok=True)
    with perq_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(perq[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in perq:
            w.writerow(row)

    def mean(values):
        return (sum(values)/len(values)) if values else 0.0

    avg_em_strict  = mean([x["EM_strict"] for x in perq])
    avg_em_numbers = mean([x["EM_numbers"] for x in perq])
    avg_f1         = mean([x["F1"] for x in perq])
    avg_rL         = mean([x["ROUGE-L"] for x in perq])

    summary = {
        "n_questions": len(perq),
        "avg_EM_strict": round(avg_em_strict,4),
        "avg_EM_numbers": round(avg_em_numbers,4),
        "avg_F1": round(avg_f1,4),
        "avg_ROUGE-L": round(avg_rL,4)
    }

    summary_json = Path(DEFAULT_OUT_DIR) / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = Path(DEFAULT_OUT_DIR) / "report.md"
    lines = [
        "# Évaluation qualité des réponses (RAG) — fidèle",
        f"- **Nombre de questions**        : {summary['n_questions']}",
        f"- **Exact Match (strict, moy.)** : {summary['avg_EM_strict']}",
        f"- **Exact Match (numérique)**    : {summary['avg_EM_numbers']}",
        f"- **F1 token (moy.)**            : {summary['avg_F1']}",
        f"- **ROUGE-L (moy.)**             : {summary['avg_ROUGE-L']}",
        "",
        "## Détails par question",
    ]
    for r in perq:
        lines.append(
            f"- **{r['qid']}** — EM_strict: {r['EM_strict']} | "
            f"EM_numbers: {r['EM_numbers']} | F1: {r['F1']} | ROUGE-L: {r['ROUGE-L']}"
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return perq_csv, summary_json, report_md


# =========================
# Gold par défaut (DocFinancier)
# =========================
DEFAULT_GOLD = [
  {"qid":"q001","doc_id":"DocFinancier","question":"Quel est le CA total à fin novembre 2024 (tous réseaux confondus) ?","answer_gold":"4 830 MMAD"},
  {"qid":"q002","doc_id":"DocFinancier","question":"Quel est le CA total à fin novembre 2023 (tous réseaux confondus) ?","answer_gold":"4 586 MMAD"},
  {"qid":"q003","doc_id":"DocFinancier","question":"Quelle est la variation du CA de la branche Vie entre fin novembre 2023 et fin novembre 2024 ?","answer_gold":"+152 MMAD (16,5%)"},
  {"qid":"q004","doc_id":"DocFinancier","question":"Quelle est la variation du CA de la branche Incendie & Multirisques entre fin novembre 2023 et fin novembre 2024 ?","answer_gold":"+61 MMAD (23,3%)"},
  {"qid":"q005","doc_id":"DocFinancier","question":"Quel est le CA Auto Mono à fin novembre 2024 (tous réseaux confondus) ?","answer_gold":"1 367 MMAD"},
  {"qid":"q006","doc_id":"DocFinancier","question":"Quel est le CA Auto Flotte à fin novembre 2024 (tous réseaux confondus) ?","answer_gold":"239 MMAD"},
  {"qid":"q007","doc_id":"DocFinancier","question":"Quel est le TRO versus budget en Auto global à fin novembre 2024 ?","answer_gold":"90%"},
  {"qid":"q008","doc_id":"DocFinancier","question":"Quelle est la croissance du réseau exclusif à fin novembre 2024 (en MMAD et en %) ?","answer_gold":"+144 MMAD (8,7%)"},
  {"qid":"q009","doc_id":"DocFinancier","question":"Quelle est la sinistralité globale du réseau exclusif ?","answer_gold":"71%"},
  {"qid":"q010","doc_id":"DocFinancier","question":"Quel est le total Auto (Mono+Flotte) à fin novembre 2024 (tous réseaux) ?","answer_gold":"1 606 MMAD"}
]

def write_default_gold(eval_dir: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    qa_path = eval_dir / "qa_gold.jsonl"
    with qa_path.open("w", encoding="utf-8") as f:
        for item in DEFAULT_GOLD:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return qa_path


# =========================
# Exemple d’utilisation
# =========================
if __name__ == "__main__":
    # ---- Paramètres rapidement modifiables ici ----
    pdf = "pipeline/DocFinancier.pdf"           # <- adapte le chemin si besoin
    chunk_size = 2048
    default_top_k = 3
    verbose = True

    tmp_dir = DEFAULT_TMP_DIR
    report_dir = DEFAULT_REPORT_DIR
    results_csv = DEFAULT_RESULTS_CSV
    runs_dir = DEFAULT_RUNS_DIR
    eval_dir = DEFAULT_EVAL_DIR

    # 1) Prépare un gold par défaut (modifiable)
    qa_gold_path = write_default_gold(Path(eval_dir))

    # 2) Questions batch (avec qid pour relier au gold)
    questions = [
        {"qid":"q001","type": "Valeur", "text": DEFAULT_GOLD[0]["question"]},
        {"qid":"q002","type": "Valeur", "text": DEFAULT_GOLD[1]["question"]},
        {"qid":"q003","type": "Valeur", "text": DEFAULT_GOLD[2]["question"]},
        {"qid":"q004","type": "Valeur", "text": DEFAULT_GOLD[3]["question"]},
        {"qid":"q005","type": "Valeur", "text": DEFAULT_GOLD[4]["question"]},
        {"qid":"q006","type": "Valeur", "text": DEFAULT_GOLD[5]["question"]},
        {"qid":"q007","type": "Autre",  "text": DEFAULT_GOLD[6]["question"]},
        {"qid":"q008","type": "Autre",  "text": DEFAULT_GOLD[7]["question"]},
        {"qid":"q009","type": "Autre",  "text": DEFAULT_GOLD[8]["question"]},
        {"qid":"q010","type": "Valeur", "text": DEFAULT_GOLD[9]["question"]},
    ]

    # 3) Lancer pipeline + logging CSV/JSONL + évaluation (fidèle) avec affichage scores
    pipeline = QwenPDFPipeline(
        chunk_size=chunk_size,
        default_top_k=default_top_k,
        temp_dir=tmp_dir,
        report_dir=report_dir,
        csv_path=results_csv,
        runs_dir=runs_dir,
        verbose=verbose,
    )

    rows = pipeline.run_batch(pdf, questions, batch_id="demo_multitype", gold_path=str(qa_gold_path))

    print("\n=== RÉCAP CSV (dernière exécution) ===")
    print(f"Fichier : {pipeline.csv_path}")
    for r in rows:
        print(f"- {r.get('question_type','?')}: {r.get('status','?')} | answer_len={r.get('answer_len','?')} | best_rank={r.get('best_rank','?')}")

    print("\n✅ Évaluation écrite dans:")
    print(f"- {DEFAULT_OUT_DIR}/answers_eval.csv")
    print(f"- {DEFAULT_OUT_DIR}/summary.json")
    print(f"- {DEFAULT_OUT_DIR}/report.md")
