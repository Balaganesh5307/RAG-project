import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -----------------------------------
# Embedding model
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# ChromaDB
# -----------------------------------

vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)

# BM25 in-memory cache by document_id
_bm25_cache = {}


def _get_bm25_for_doc(document_id):
    if document_id in _bm25_cache:
        return _bm25_cache[document_id]

    data = vector_store.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not docs:
        return None

    tokenized = [re.findall(r"\w+", d.lower()) for d in docs]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache[document_id] = (bm25, docs, metadatas)
    return _bm25_cache[document_id]


def _is_reference_chunk(text):
    """Detect bibliography / pure citation chunks so they don't crowd out real content."""
    citation_patterns = [
        r'\b(?:19|20)\d{2}[a-z]?\b.*?\b(?:In:|Proceedings|arXiv|doi|pp\s*\d+|vol\b)',
        r'arXiv:\d+\.\d+',
        r'https?://doi\.org/'
    ]
    matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in citation_patterns)
    return matches >= 2


# -----------------------------------
# Multi-query variant generation
# -----------------------------------

def _generate_query_variants(requirement, question=None):
    """
    Generate query variants for dense retrieval to maximize recall,
    especially for acronyms and definition questions.
    """
    queries = [requirement]

    if question and question.strip().lower() != requirement.strip().lower():
        queries.append(question.strip())

    # Check for acronyms with expansions like "RAG (Retrieval-Augmented Generation)"
    acronym_match = re.search(r"(\b[A-Za-z0-9]{2,10}\b)\s*\(([^)]+)\)", requirement)
    if acronym_match:
        acronym, expansion = acronym_match.group(1), acronym_match.group(2)
        queries.append(f"What is {expansion}?")
        queries.append(f"{expansion} ({acronym})")
        queries.append(f"known as {expansion} ({acronym})")
        queries.append(f"{expansion} definition and overview")

    # If asking for definition or overview
    def_match = re.search(r"(?:definition|meaning|overview|purpose)\s+of\s+([^,.]+)", requirement, re.IGNORECASE)
    if def_match:
        subject = def_match.group(1).strip()
        queries.append(f"What is {subject}?")
        queries.append(f"known as {subject}")
        queries.append(f"{subject} definition and overview")

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        cleaned = q.strip()
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique_queries.append(cleaned)

    return unique_queries


# -----------------------------------
# Requirement-wise retrieval
# -----------------------------------

def retrieve_for_requirement(
    requirement,
    document_id,
    k=8,
    question=None
):
    query_variants = _generate_query_variants(requirement, question)

    candidates = {}

    # 1. Dense retrieval across query variants with Reciprocal Rank Fusion
    for q in query_variants:
        try:
            results = vector_store.similarity_search_with_score(
                q,
                k=k,
                filter={
                    "document_id": document_id
                }
            )
            for rank, (doc, score) in enumerate(results):
                cid = doc.metadata.get("chunk_id", doc.page_content)
                rrf = 1.0 / (60 + rank)
                if cid not in candidates:
                    candidates[cid] = {"doc": doc, "score": score, "rrf": rrf}
                else:
                    candidates[cid]["rrf"] += rrf
                    if score < candidates[cid]["score"]:
                        candidates[cid]["score"] = score
        except Exception:
            continue

    # 2. BM25 keyword retrieval
    try:
        bm25_data = _get_bm25_for_doc(document_id)
        if bm25_data:
            bm25, docs, metadatas = bm25_data
            bm25_queries = [requirement]
            if question:
                bm25_queries.append(question)
            acronym_match = re.search(r"(\b[A-Za-z0-9]{2,10}\b)\s*\(([^)]+)\)", requirement)
            if acronym_match:
                acronym, expansion = acronym_match.group(1), acronym_match.group(2)
                bm25_queries.append(f"known as {expansion} {acronym}")
                bm25_queries.append(f"{expansion} definition")

            for bq in bm25_queries:
                tokens = re.findall(r"\w+", bq.lower())
                if not tokens:
                    continue
                scores = bm25.get_scores(tokens)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
                for rank, idx in enumerate(top_indices):
                    d_text = docs[idx]
                    cid = metadatas[idx].get("chunk_id", d_text)
                    rrf = 1.0 / (60 + rank)
                    if cid in candidates:
                        candidates[cid]["rrf"] += rrf
                    else:
                        doc = Document(page_content=d_text, metadata=metadatas[idx])
                        candidates[cid] = {"doc": doc, "score": 1.0, "rrf": rrf}
    except Exception:
        pass

    # Fallback to single query if candidates empty
    if not candidates:
        return vector_store.similarity_search_with_score(
            requirement,
            k=k,
            filter={
                "document_id": document_id
            }
        )

    # 3. Filter out references
    valid_candidates = []
    for cid, item in candidates.items():
        doc = item["doc"]
        if _is_reference_chunk(doc.page_content):
            continue
        valid_candidates.append((doc, item["score"], item["rrf"]))

    if not valid_candidates:
        valid_candidates = [(item["doc"], item["score"], item["rrf"]) for item in candidates.values()]

    # Sort by RRF descending (and score ascending)
    valid_candidates.sort(key=lambda x: (-x[2], x[1]))

    return [(doc, score) for doc, score, _ in valid_candidates[:k]]