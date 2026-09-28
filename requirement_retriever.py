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


# -----------------------------------
# Requirement-wise retrieval
# -----------------------------------

def retrieve_for_requirement(
    requirement,
    document_id,
    k=3
):

    results = vector_store.similarity_search_with_score(
        requirement,
        k=k,
        filter={
            "document_id": document_id
        }
    )

    return results