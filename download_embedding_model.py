from sentence_transformers import SentenceTransformer


# Model name from Hugging Face
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")


# Test sentence
text = "Schema is the overall design of a database."


# Generate embedding
embedding = model.encode(text)


print("\nEmbedding created successfully!")
print("Vector dimensions:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])