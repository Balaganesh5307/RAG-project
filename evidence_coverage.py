from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from requirement_extractor import extract_requirements
from evidence_verifier import verify_requirement


load_dotenv()


# -----------------------------------
# 1. Load embedding model
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# 2. Connect to ChromaDB
# -----------------------------------

vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


# -----------------------------------
# 3. User question
# -----------------------------------

question = "What are the ACID properties and database indexing?"


# -----------------------------------
# 4. Extract requirements
# -----------------------------------

requirements_text = extract_requirements(question)

print("Question:")
print(question)

print("\n================ REQUIREMENTS ================\n")
print(requirements_text)


# -----------------------------------
# 5. Convert numbered list to Python list
# -----------------------------------

requirements = []

for line in requirements_text.splitlines():

    line = line.strip()

    if not line:
        continue

    if line[0].isdigit():

        requirement = line.split(".", 1)[1].strip()

        requirements.append(requirement)


print("\nParsed requirements:")
for requirement in requirements:
    print("-", requirement)


# -----------------------------------
# 6. Retrieve evidence
# -----------------------------------

results = vector_store.similarity_search(
    question,
    k=3
)


# -----------------------------------
# 7. Build evidence text
# -----------------------------------

evidence_parts = []

for i, document in enumerate(results):

    evidence_parts.append(
        f"""
Evidence {i + 1}
Source: {document.metadata.get("source")}
Page: {document.metadata.get("page", 0) + 1}

{document.page_content}
"""
    )


evidence = "\n".join(evidence_parts)


# -----------------------------------
# 8. Verify every requirement
# -----------------------------------

verification_results = []

for requirement in requirements:

    result = verify_requirement(
        requirement,
        evidence
    )

    verification_results.append({
        "requirement": requirement,
        "status": result["status"],
        "evidence": result["evidence"],
        "reason": result["reason"]
    })


# -----------------------------------
# 9. Display verification results
# -----------------------------------

print("\n================ VERIFICATION ================\n")

for result in verification_results:

    print("Requirement:")
    print(result["requirement"])

    print("Status:")
    print(result["status"])

    print("Evidence:")
    print(result["evidence"])

    print("Reason:")
    print(result["reason"])

    print("\n------------------------------")


# -----------------------------------
# 10. Calculate evidence coverage
# -----------------------------------

supported_count = 0

for result in verification_results:

    if result["status"] == "SUPPORTED":
        supported_count += 1


total_requirements = len(verification_results)


if total_requirements > 0:

    coverage = supported_count / total_requirements

else:

    coverage = 0


print("\n================ EVIDENCE COVERAGE ================\n")

print("Supported requirements:", supported_count)
print("Total requirements:", total_requirements)
print("Evidence coverage:", round(coverage, 4))