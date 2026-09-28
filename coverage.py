from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to ChromaDB
vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


# Question
question = "What are the ACID properties?"


# Retrieve evidence
results = vector_store.similarity_search(
    question,
    k=3
)


# Information requirements for this test
requirements = [
    "Atomicity",
    "Consistency",
    "Isolation",
    "Durability"
]


# Combine retrieved evidence
evidence_texts = [
    document.page_content
    for document in results
]


# Create embeddings for evidence
evidence_vectors = embeddings.embed_documents(
    evidence_texts
)


print("Question:")
print(question)

print("\n================ COVERAGE EVALUATION ================")


covered_requirements = []


for requirement in requirements:

    # Embed the requirement
    requirement_vector = embeddings.embed_query(
        requirement
    )


    # Compare requirement against every evidence chunk
    similarities = []

    for evidence_vector in evidence_vectors:

        similarity = cosine_similarity(
            [requirement_vector],
            [evidence_vector]
        )[0][0]

        similarities.append(similarity)


    # Find the strongest evidence for this requirement
    best_similarity = max(similarities)


    print("\nRequirement:", requirement)

    print(
        "Best evidence similarity:",
        round(best_similarity, 4)
    )


    # No threshold yet.
    # We are only measuring the signal.