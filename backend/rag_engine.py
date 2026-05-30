import os
import json
import re
import numpy as np
import requests
from pypdf import PdfReader
import google.generativeai as genai
from typing import List, Dict, Any, Tuple, Optional

class PDFProcessor:
    """Handles PDF loading, text extraction, cleaning, and semantic chunking."""
    
    @staticmethod
    def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text page by page from the PDF, cleaning basic layout artifacts."""
        pages_data = []
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = page.extract_text()
            if not text:
                text = ""
                
            # Basic cleaning: remove excessive whitespace and running headers/footers if matched
            lines = text.split("\n")
            cleaned_lines = []
            for line in lines:
                line_strip = line.strip()
                # Ignore empty lines
                if not line_strip:
                    continue
                # Skip standard ISO page numbering/header headers if they clutter context
                if re.match(r"^ISO\s+9001:2015\(E\)", line_strip, re.IGNORECASE):
                    continue
                cleaned_lines.append(line_strip)
                
            cleaned_text = "\n".join(cleaned_lines)
            pages_data.append({
                "page_number": page_num,
                "raw_text": text,
                "cleaned_text": cleaned_text
            })
            
        return pages_data

    @staticmethod
    def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
        """
        Chunks the page text using a sliding window.
        Tracks source page, clause titles, and sequence.
        """
        chunks = []
        chunk_id = 0
        
        # Regex to capture major clauses (e.g., "4 Context of the organization", "5.1 Leadership and commitment")
        clause_pattern = re.compile(r"^(?:Clause\s+)?(\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z0-9\s,\-\(\)/]+)")

        for page in pages_data:
            text = page["cleaned_text"]
            page_num = page["page_number"]
            
            # Identify any clauses appearing on this page to enrich chunk metadata
            clauses_found = []
            for line in text.split("\n"):
                match = clause_pattern.match(line)
                if match:
                    clause_num, clause_name = match.groups()
                    clauses_found.append(f"{clause_num} {clause_name.strip()}")
            
            # Core sliding window chunking
            start = 0
            text_len = len(text)
            
            if text_len == 0:
                continue
                
            while start < text_len:
                end = min(start + chunk_size, text_len)
                
                # Try to adjust boundary to end at a sentence or paragraph rather than mid-word
                if end < text_len:
                    # Look for sentence boundary (.!?) or newline in the last 60 chars of the chunk
                    boundary = -1
                    for separator in [". ", "\n", " "]:
                        pos = text.rfind(separator, end - 60, end)
                        if pos != -1:
                            boundary = pos + len(separator)
                            break
                    if boundary != -1:
                        end = boundary
                
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 50:  # Skip trivial chunks
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "page": page_num,
                        "clauses": clauses_found if clauses_found else ["General / Contextual"]
                    })
                    chunk_id += 1
                
                start += chunk_size - overlap
                if start >= text_len or end >= text_len:
                    break
                    
        return chunks


class VectorStore:
    """Lightweight in-memory vector database with JSON serialization and cosine similarity."""
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # NumPy 2D array of shape (N, D)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Adds text chunks along with their computed embedding vectors."""
        self.chunks.extend(chunks)
        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Performs cosine similarity search against stored chunks."""
        if not self.chunks or self.embeddings is None:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        
        # Compute cosine similarity
        dot_products = np.dot(self.embeddings, q_vec)
        norms_docs = np.linalg.norm(self.embeddings, axis=1)
        norm_q = np.linalg.norm(q_vec)
        
        # Prevent division by zero
        norms_docs[norms_docs == 0] = 1e-10
        if norm_q == 0:
            norm_q = 1e-10
            
        similarities = dot_products / (norms_docs * norm_q)
        
        # Get top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
            
        return results

    def save(self, filepath: str):
        """Serializes the index to a single JSON-compatible file for fast loading."""
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else []
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> bool:
        """Loads an existing serialized index from disk."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            if data["embeddings"]:
                self.embeddings = np.array(data["embeddings"], dtype=np.float32)
            else:
                self.embeddings = None
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False


class RAGEngine:
    """Orchestrates the entire RAG pipeline from ingestion to query answering."""
    
    def __init__(self, api_key: str = "", provider: str = "gemini"):
        self.api_key = api_key
        self.provider = provider.lower()
        
        # Configure Gemini if provider is active
        if self.provider == "gemini" and api_key:
            genai.configure(api_key=api_key)
            
        self.vector_store = VectorStore()
        
        # Isolate database file names by provider to prevent matrix shape conflicts!
        if self.provider == "gemini":
            self.index_name = "vector_store_gemini.json"
        elif self.provider == "cohere":
            self.index_name = "vector_store_cohere.json"
        else:
            self.index_name = "vector_store_ollama.json"
        self.index_file = os.path.join(os.path.dirname(__file__), self.index_name)

    def _get_embeddings_cohere(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """Invokes the Cohere embeddings REST service (embed-english-v3.0)."""
        url = "https://api.cohere.com/v1/embed"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "texts": texts,
            "model": "embed-english-v3.0",
            "input_type": input_type
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            raise RuntimeError(f"Cohere embedding failed: {e}")

    def _query_cohere(self, prompt: str, system_instruction: str, model_name: str, temperature: float) -> str:
        """Invokes the Cohere chat REST service (v1/chat)."""
        url = "https://api.cohere.com/v1/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "message": prompt,
            "model": model_name,
            "preamble": system_instruction,
            "temperature": temperature
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["text"]
        except Exception as e:
            raise RuntimeError(f"Cohere chat generation failed: {e}")

    def _get_embeddings_ollama(self, texts: List[str]) -> List[List[float]]:
        """Invokes the local Ollama embeddings REST service (nomic-embed-text)."""
        url = "http://127.0.0.1:11434/api/embeddings"
        embeddings = []
        for text in texts:
            try:
                response = requests.post(url, json={
                    "model": "nomic-embed-text",
                    "prompt": text
                }, timeout=90)
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except Exception as e:
                raise RuntimeError(
                    f"Ollama local embedding failed. Make sure you pulled the embedding model first "
                    f"by running 'ollama pull nomic-embed-text' in your terminal. Error: {e}"
                )
        return embeddings

    def _query_ollama(self, prompt: str, system_instruction: str, model_name: str, temperature: float) -> str:
        """Invokes the local Ollama chat REST service."""
        url = "http://127.0.0.1:11434/api/chat"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": temperature
            },
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"Ollama chat generation failed. Make sure you have downloaded the local model "
                f"by running 'ollama pull {model_name}' and Ollama is active. Error: {e}"
            )

    def _rewrite_query(self, query: str, ollama_model: str, gemini_model: str, cohere_model: str) -> str:
        """Uses the active LLM provider to rewrite the query to be optimized for semantic search."""
        system_instruction = (
            "You are a search query optimizer. Your job is to rewrite a user query about ISO 9001:2015 "
            "into a direct, keyword-rich search query optimized for finding relevant standard clauses in a PDF document. "
            "Keep the query focused and brief. Return ONLY the rewritten query text, with absolutely no preamble, introductory text, "
            "or enclosing quotation marks."
        )
        prompt = f"Optimize this user query for vector database semantic search: {query}"
        
        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(
                    model_name=gemini_model,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1}
                )
                rewritten = response.text.strip()
            elif self.provider == "cohere":
                rewritten = self._query_cohere(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=cohere_model,
                    temperature=0.1
                ).strip()
            else:
                rewritten = self._query_ollama(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=ollama_model,
                    temperature=0.1
                ).strip()
            
            # Clean up the output if the model added quotation marks
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1].strip()
            elif rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1].strip()
                
            return rewritten
        except Exception as e:
            print(f"Warning: Query rewrite failed: {e}")
            return query  # Fallback to the original query if LLM rewrite fails

    def ingest_pdf(self, pdf_path: str, force_rebuild: bool = False) -> Tuple[int, bool]:
        """
        Parses, chunks, embeds, and indexes a PDF.
        Loads from serialized file if it exists and force_rebuild is False.
        Returns: (number of chunks indexed, loaded_from_cache_flag)
        """
        if not force_rebuild and self.vector_store.load(self.index_file):
            return len(self.vector_store.chunks), True
            
        # 1. Extract and Clean Text
        pages_data = PDFProcessor.extract_text_by_page(pdf_path)
        
        # 2. Chunk text
        chunks = PDFProcessor.chunk_text(pages_data)
        if not chunks:
            return 0, False
            
        # 3. Generate Embeddings
        embeddings = []
        
        if self.provider == "gemini":
            # Generate Gemini Embeddings in rate-limit safe batches (max 100 RPM limit on free tier)
            import time
            batch_size = 40
            for i in range(0, len(chunks), batch_size):
                if i > 0:
                    # Sleep to stay well under the 100 requests-per-minute limit
                    time.sleep(25)
                batch = chunks[i:i+batch_size]
                texts = [c["text"] for c in batch]
                response = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=texts,
                    task_type="retrieval_document"
                )
                embeddings.extend(response["embedding"])
        elif self.provider == "cohere":
            # Generate Cohere Embeddings
            batch_size = 90
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [c["text"] for c in batch]
                batch_embeddings = self._get_embeddings_cohere(texts, input_type="search_document")
                embeddings.extend(batch_embeddings)
        else:
            # Generate Ollama Local Embeddings
            texts = [c["text"] for c in chunks]
            embeddings = self._get_embeddings_ollama(texts)
            
        # 4. Add to VectorStore and Save
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save(self.index_file)
        
        return len(chunks), False

    def query(self, user_query: str, top_k: int = 5, temperature: float = 0.2, ollama_model: str = "llama3", gemini_model: str = "gemini-2.5-flash", cohere_model: str = "command-r-08-2024") -> Dict[str, Any]:
        """
        Runs the RAG query:
        1. Embeds the user query (Gemini or Ollama).
        2. Retrieves the most relevant chunks from the corresponding vector store.
        3. Formulates context.
        4. Generates response (Gemini or local Ollama).
        """
        if not self.vector_store.chunks:
            raise ValueError("No chunks have been indexed. Please ingest the PDF document first.")
            
        # 1. Embed query (with bounded retry if top score < 0.5)
        current_query = user_query
        retrieved = []
        max_retries = 2
        attempts_log = []
        
        for attempt in range(max_retries + 1):
            if self.provider == "gemini":
                query_resp = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=current_query,
                    task_type="retrieval_query"
                )
                query_embedding = query_resp["embedding"]
            elif self.provider == "cohere":
                query_embedding = self._get_embeddings_cohere([current_query], input_type="search_query")[0]
            else:
                query_embedding = self._get_embeddings_ollama([current_query])[0]
            
            # 2. Retrieve top matching chunks
            retrieved = self.vector_store.query(query_embedding, top_k=top_k)
            
            # Check similarity score of the top retrieved chunk
            top_score = retrieved[0][1] if retrieved else 0.0
            attempts_log.append({
                "attempt": attempt,
                "query": current_query,
                "top_score": top_score
            })
            
            if top_score >= 0.5 or attempt == max_retries:
                # Similarity is good enough, or we've hit max retries
                break
                
            # Otherwise, rewrite query for retry
            print(f"Top similarity score ({top_score:.3f}) < 0.5. Rewriting query and retrying (Attempt {attempt + 1}/{max_retries})...")
            current_query = self._rewrite_query(
                query=current_query,
                ollama_model=ollama_model,
                gemini_model=gemini_model,
                cohere_model=cohere_model
            )
        
        # 3. Build Context String
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved):
            clauses_str = ", ".join(chunk["clauses"])
            context_parts.append(
                f"[Source #{i+1} | Page {chunk['page']} | Clauses: {clauses_str} | Similarity: {score:.3f}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # 4. Define Auditor System Instruction
        system_instruction = (
            "You are a professional Lead QMS Auditor and subject matter expert in the ISO 9001:2015 standard.\n"
            "Your objective is to provide highly accurate, structured, and helpful advice based strictly on the "
            "standard requirements contained in the provided PDF text.\n\n"
            "INSTRUCTIONS FOR YOUR RESPONSE:\n"
            "1. Ground your answers thoroughly in the provided Context. Only discuss information that is directly supported by the context.\n"
            "2. If the context does not contain enough information to answer a question, state this clearly and specify what details are missing.\n"
            "3. Cite your sources using bracketed notation like [Source #1], referencing the page numbers and clauses as given in the context headers.\n"
            "4. Organize your answers using clear markdown: use bullet points, bold text for emphasis, tables for comparisons, and checklist boxes for audit steps.\n"
            "5. Keep a highly professional, auditing-focused, and objective tone."
        )
        
        # 5. Build prompt
        prompt = (
            f"Context from the ISO 9001:2015 standard:\n"
            f"{context}\n\n"
            f"User Question: {user_query}\n\n"
            f"Please formulate a precise, structured auditor response, complete with page and source citations."
        )
        
        # 6. Generate Response
        if self.provider == "gemini":
            model = genai.GenerativeModel(
                model_name=gemini_model,
                system_instruction=system_instruction
            )
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            answer_text = response.text
        elif self.provider == "cohere":
            answer_text = self._query_cohere(
                prompt=prompt,
                system_instruction=system_instruction,
                model_name=cohere_model,
                temperature=temperature
            )
        else:
            answer_text = self._query_ollama(
                prompt=prompt,
                system_instruction=system_instruction,
                model_name=ollama_model,
                temperature=temperature
            )
        
        return {
            "answer": answer_text,
            "sources": [
                {
                    "source_id": i + 1,
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "clauses": chunk["clauses"],
                    "score": score
                }
                for i, (chunk, score) in enumerate(retrieved)
            ],
            "query_attempts": attempts_log
        }
