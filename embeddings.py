from langchain_huggingface import HuggingFaceEmbeddings


# Model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Create LangChain embedding object
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


# Test text
text = "Schema is the overall design of a database."


# Create embedding
vector = embeddings.embed_query(text)


print("Embedding created successfully!")

print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])