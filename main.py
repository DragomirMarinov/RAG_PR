import os
import glob
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

console = Console()

class RAGLite:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 300, overlap: int = 50):
        console.print(f"[bold blue]Loading embedding model '{model_name}'...[/bold blue]")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])

    def _chunk_text(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Splits text into sliding window chunks."""
        words = text.split()
        chunks = []
        if not words:
            return chunks

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "source": os.path.basename(file_path),
                "text": chunk_text
            })
            if end >= len(words):
                break
            start += (self.chunk_size - self.overlap)
        return chunks

    def ingest_docs(self, docs_folder: str = "docs"):
        """Reads all txt/md files in docs_folder and creates embeddings."""
        file_paths = glob.glob(os.path.join(docs_folder, "*.txt")) + glob.glob(os.path.join(docs_folder, "*.md"))
        
        if not file_paths:
            console.print(f"[bold yellow]No .txt or .md files found in '{docs_folder}/'. Please add some files first![/bold yellow]")
            return False

        all_chunks = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                file_chunks = self._chunk_text(content, path)
                all_chunks.extend(file_chunks)

        if not all_chunks:
            console.print("[bold yellow]No readable text found in documents.[/bold yellow]")
            return False

        self.chunks = all_chunks
        console.print(f"[cyan]Ingested {len(file_paths)} document(s) into {len(self.chunks)} text chunks.[/cyan]")
        
        # Generate embeddings
        texts = [c["text"] for c in self.chunks]
        console.print("[bold blue]Generating vector embeddings...[/bold blue]")
        raw_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Normalize embeddings for cosine similarity via dot product
        self.embeddings = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        console.print("[bold green]Vector index built successfully![/bold green]\n")
        return True

    def search(self, query: str, top_k: int = 3):
        """Vector similarity lookup using normalized dot product."""
        if self.embeddings.size == 0:
            console.print("[bold red]No index loaded. Ingest documents first.[/bold red]")
            return

        query_vec = self.model.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Cosine similarity = dot product of normalized vectors
        scores = np.dot(self.embeddings, query_vec.T).squeeze()
        
        # Handle single chunk edge case
        if scores.ndim == 0:
            top_indices = [0]
            scores = np.array([scores])
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]

        table = Table(title=f"Query Results for: '{query}'", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Score", width=8)
        table.add_column("Source File", width=18)
        table.add_column("Matching Excerpt", style="white")

        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            chunk = self.chunks[idx]
            table.add_row(
                str(rank),
                f"{score:.4f}",
                chunk["source"],
                chunk["text"][:250] + ("..." if len(chunk["text"]) > 250 else "")
            )

        console.print(table)


def main():
    console.print(Panel.fit("[bold green]RAGLite: Local Semantic Document Search Engine[/bold green]"))
    rag = RAGLite()

    # Step 1: Ingest
    if not rag.ingest_docs("docs"):
        return

    # Step 2: Interactive loop
    while True:
        try:
            query = Prompt.ask("\n[bold yellow]Enter search query[/bold yellow] (or type 'exit')")
            if query.strip().lower() in ["exit", "quit", "q"]:
                console.print("[bold cyan]Goodbye![/bold cyan]")
                break
            if not query.strip():
                continue
            rag.search(query, top_k=3)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()