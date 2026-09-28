from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


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


# Step 4: Retrieve evidence
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


context = "\n".join(context_parts)


# Step 6: Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are DocLens AI, an evidence-first document assistant.

Answer the user's question using ONLY the provided evidence.

If the evidence does not contain enough information to answer the question,
say that the information is not available in the provided document.

Do not invent facts.

User Question:
{question}

Evidence:
{context}

Answer:
"""
)


# Step 7: Create final prompt
final_prompt = prompt.invoke({
    "question": question,
    "context": context
})


# Step 8: Display the prompt
print(final_prompt.to_string())