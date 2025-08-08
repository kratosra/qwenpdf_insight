from pipeline.cache_manager import EmbeddingCacheManager
from pipeline.extract.extract_with_docling import pdf_to_markdown, chunk_markdown
from pipeline.embedding.qwen_embedding import embed_chunks_qwen3, build_faiss_index, retrieve_top_k_chunks
from pipeline.generation.generate_qwen_answer import generate_answer_qwen_chat_format
from pathlib import Path

class QwenPDFPipeline:
    def __init__(self, chunk_size=2048, top_k=3, temp_dir="data/tmp", cache_dir="data/cache", verbose=True):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.cache = EmbeddingCacheManager(cache_dir)

    def log(self, msg):
        if self.verbose:
            print(f"[QwenPDFPipeline] {msg}")

    def extract_chunks_with_cache(self, pdf_path: Path):
        cached = self.cache.load_cache(pdf_path)
        if cached:
            self.log(" Chargement depuis le cache")
            return cached

        self.log("Extraction Markdown...")
        md_path = pdf_to_markdown(pdf_path, output_dir=self.temp_dir / "markdown")

        self.log("Découpage en chunks...")
        chunks = chunk_markdown(md_path, chunk_size=self.chunk_size)

        self.log("Embedding des chunks...")
        embeddings = np.array(embed_chunks_qwen3(chunks))

        self.log("Construction index...")
        index = build_faiss_index(embeddings)

        self.cache.save_cache(pdf_path, chunks, embeddings, index)
        return chunks, embeddings, index

    def run(self, pdf_path: str, question: str):
        chunks, embeddings, index = self.extract_chunks_with_cache(Path(pdf_path))
        top_chunks = retrieve_top_k_chunks(question, chunks, embeddings, index, top_k=self.top_k)
        answer = generate_answer_qwen_chat_format(top_chunks, question)
        return answer, top_chunks
