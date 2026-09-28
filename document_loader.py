from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("sample.pdf")

documents = loader.load()

print("Number of documents:", len(documents))

for document in documents[:3]:
    print("\n--- DOCUMENT ---")
    print("Content:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)