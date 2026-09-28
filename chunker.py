from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Load PDF
loader = PyMuPDFLoader("sample.pdf")
documents = loader.load()


# Step 2: Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# Step 3: Split documents
chunks = text_splitter.split_documents(documents)


# Step 4: Display results
print("Number of original documents:", len(documents))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n==============================")
    print(f"CHUNK {i + 1}")
    print("==============================")

    print("Content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)