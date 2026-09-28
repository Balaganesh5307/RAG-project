from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from document_fingerprint import calculate_file_hash
from document_ingestion import ingest_document


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


def find_document_by_hash(file_hash):

    result = vector_store.get(
        where={
            "file_hash": file_hash
        },
        limit=1
    )

    ids = result.get("ids", [])

    if not ids:
        return None

    metadatas = result.get("metadatas", [])

    if not metadatas:
        return None

    metadata = metadatas[0]

    return {
        "document_id": metadata.get("document_id"),
        "file_hash": metadata.get("file_hash")
    }


def store_document(file_path):

    # 1. Calculate hash BEFORE PDF ingestion
    file_hash = calculate_file_hash(file_path)

    # 2. Check whether this exact document already exists
    existing_document = find_document_by_hash(file_hash)

    if existing_document is not None:
        return {
            "document_id": existing_document["document_id"],
            "file_hash": existing_document["file_hash"],
            "pages": 0,
            "chunks": 0,
            "duplicate": True
        }

    # 3. New document → load and process PDF
    ingestion_result = ingest_document(file_path)

    document_id = ingestion_result["document_id"]
    documents = ingestion_result["documents"]
    chunks = ingestion_result["chunks"]

    # 4. Store chunks and embeddings
    vector_store.add_documents(chunks)

    return {
        "document_id": document_id,
        "file_hash": file_hash,
        "pages": len(documents),
        "chunks": len(chunks),
        "duplicate": False
    }