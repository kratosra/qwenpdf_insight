from pathlib import Path
from typing import List, Tuple, Optional
import traceback
import sys
from pathlib import Path

# Ajoute le dossier racine (qwenpdf_insight) au PYTHONPATH dynamiquement
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.extract.extract_with_docling import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format

class QwenPDFPipeline:
    def __init__(
        self,
        chunk_size: int = 2048,
        top_k: int = 3,
        temp_dir: str = "data/tmp",
        verbose: bool = True
    ):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            print(f"[QwenPDFPipeline] {message}")

    def extract_chunks(self, pdf_path: Path) -> List[str]:
        self.log("Extraction Markdown...")
        md_path = pdf_to_markdown(pdf_path, output_dir=self.temp_dir / "markdown")

        self.log("Découpage en chunks...")
        chunks = chunk_markdown(md_path, chunk_size=self.chunk_size, output_chunks_path=self.temp_dir / "chunks.txt")
        return chunks

    def retrieve_chunks(self, chunks: List[str], question: str) -> List[str]:
        self.log("Embedding des chunks...")
        embeddings = embed_chunks_qwen3(chunks)

        self.log("Indexation FAISS...")
        index = build_faiss_index(embeddings)

        self.log("Recherche des chunks les plus pertinents...")
        return retrieve_top_k_chunks(question, chunks, embeddings, index, top_k=self.top_k)

    def generate_answer(self, top_chunks: List[str], question: str) -> str:
        self.log("Génération de la réponse...")
        return generate_answer_qwen_chat_format(top_chunks, question)

    def run(self, pdf_path: str, question: str) -> Tuple[str, List[str]]:
        try:
            pdf_path = Path(pdf_path)
            chunks = self.extract_chunks(pdf_path)
            if not chunks:
                return "Aucun contenu exploitable trouvé.", []

            top_chunks = self.retrieve_chunks(chunks, question)
            answer = self.generate_answer(top_chunks, question)

            return answer, top_chunks
        except Exception as e:
            error = f"Erreur dans le pipeline : {e}\n{traceback.format_exc()}"
            self.log(error)
            return "Erreur durant l’analyse du document.", []


if __name__ == "__main__":
    pipeline = QwenPDFPipeline(chunk_size=2048, top_k=3)
    answer, context = pipeline.run("pipeline/analyse_critique_projet_qwenpdf.pdf", "Quel est le chiffre d'affaires ?")

    print("Réponse :")
    print(answer)
