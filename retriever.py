from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


document_id = "9a5a4b4a-488e-4a38-a171-07160cbc735b"

requirement = "What is the purpose of a README?"


results = vector_store.similarity_search_with_score(
    requirement,
    k=5,
    filter={
        "document_id": document_id
    }
)


print("Requirement:")
print(requirement)

print("\nDocument ID:")
print(document_id)

print("\nRetrieved chunks:")

for i, (document, score) in enumerate(results):

    print("\n==============================")
    print(f"RESULT {i + 1}")
    print("==============================")

    print("\nDistance:")
    print(score)

    print("\nPage:")
    print(document.metadata.get("page", 0) + 1)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)