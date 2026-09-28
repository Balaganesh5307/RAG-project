import os
import uuid

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_fingerprint import calculate_file_hash


def ingest_document(file_path):

    # 1. Calculate document fingerprint
    file_hash = calculate_file_hash(file_path)

    # 2. Generate unique document ID
    document_id = str(uuid.uuid4())

    # 3. Load PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # 4. Add document-level metadata
    for document in documents:
        document.metadata["document_id"] = document_id
        document.metadata["file_hash"] = file_hash
        document.metadata["source"] = os.path.abspath(file_path)

    # 5. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # 6. Add chunk IDs
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{document_id}_{index}"

    return {
        "document_id": document_id,
        "file_hash": file_hash,
        "documents": documents,
        "chunks": chunks
    }

if __name__ == "__main__":
    result = ingest_document("sample.pdf")

    print("Document ID:")
    print(result["document_id"])

    print("\nFile Hash:")
    print(result["file_hash"])

    print("\nPages:")
    print(len(result["documents"]))

    print("\nChunks:")
    print(len(result["chunks"]))

    print("\nFirst Chunk Metadata:")
    print(result["chunks"][0].metadata)