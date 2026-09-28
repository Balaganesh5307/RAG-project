from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Step 1: Load PDF
loader = PyMuPDFLoader("sample.pdf")
documents = loader.load()


# Step 2: Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# Step 3: Create chunks
chunks = text_splitter.split_documents(documents)


# Step 4: Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 5: Extract chunk text
texts = [chunk.page_content for chunk in chunks]


# Step 6: Convert all chunks into embeddings
vectors = embeddings.embed_documents(texts)


# Step 7: Display results
print("Number of chunks:", len(chunks))

print("Number of vectors:", len(vectors))

print("Vector dimensions:", len(vectors[0]))


for i, vector in enumerate(vectors):
    print(f"\nChunk {i + 1}")
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])