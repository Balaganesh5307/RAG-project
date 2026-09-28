from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Step 1: Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 2: Connect to ChromaDB
vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


# Step 3: User question
question = "What are the ACID properties?"


# Step 4: Retrieve relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)


# Step 5: Build context
context_parts = []

for i, document in enumerate(results):

    context_parts.append(
        f"""
Evidence {i + 1}
Source: {document.metadata.get("source")}
Page: {document.metadata.get("page", 0) + 1}

{document.page_content}
"""
    )


# Step 6: Combine all evidence
context = "\n".join(context_parts)


# Step 7: Display context
print("QUESTION:")
print(question)

print("\n================ CONTEXT ================")
print(context)