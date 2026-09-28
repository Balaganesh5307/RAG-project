from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 2: Define sentences
sentence_a = "Schema is the overall design of a database."

sentence_b = "What is the structure of a database?"

sentence_c = "ACID properties describe transaction guarantees."

sentence_d = "How do I make pasta?"


# Step 3: Convert sentences into vectors
vector_a = embeddings.embed_query(sentence_a)
vector_b = embeddings.embed_query(sentence_b)
vector_c = embeddings.embed_query(sentence_c)
vector_d = embeddings.embed_query(sentence_d)


# Step 4: Calculate similarity
similarity_ab = cosine_similarity([vector_a], [vector_b])[0][0]

similarity_ac = cosine_similarity([vector_a], [vector_c])[0][0]

similarity_ad = cosine_similarity([vector_a], [vector_d])[0][0]


# Step 5: Display results
print("A ↔ B:", similarity_ab)

print("A ↔ C:", similarity_ac)

print("A ↔ D:", similarity_ad)