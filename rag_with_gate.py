from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Step 1: Load embedding model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Step 2: Connect to ChromaDB
# ============================================================

vector_store = Chroma(
    collection_name="doclens_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


# ============================================================
# Step 3: Evidence Gate
# ============================================================

def check_evidence(results):

    scores = [
        score
        for document, score in results
    ]

    best_score = min(scores)

    average_score = sum(scores) / len(scores)

    # Experimental threshold
    BEST_DISTANCE_THRESHOLD = 1.70

    if best_score <= BEST_DISTANCE_THRESHOLD:

        status = "SUFFICIENT"

    else:

        status = "INSUFFICIENT"

    return status, best_score, average_score


# ============================================================
# Step 4: Automatic Evidence Coverage
# ============================================================

def check_coverage(question, results):

    # Combine retrieved evidence
    evidence_text = ""

    for document, score in results:

        evidence_text += (
            document.page_content + "\n"
        )


    # --------------------------------------------------------
    # Baseline concept extraction
    # --------------------------------------------------------
    #
    # For this learning phase, we use important terms already
    # appearing in the retrieved evidence.
    #
    # Later we can replace this with a stronger concept
    # extraction method.
    # --------------------------------------------------------

    question_words = set(
        question.lower().replace("?", "").split()
    )


    evidence_words = set(
        evidence_text.lower().split()
    )


    # Remove very common question words
    stop_words = {
        "what",
        "are",
        "is",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "how",
        "does",
        "do",
        "why",
        "which",
        "what's"
    }


    important_question_words = (
        question_words - stop_words
    )


    # Find question terms that also appear in evidence
    matched_terms = (
        important_question_words
        & evidence_words
    )


    # Calculate simple term coverage
    if len(important_question_words) > 0:

        coverage_ratio = (
            len(matched_terms)
            / len(important_question_words)
        )

    else:

        coverage_ratio = 0


    return (
        important_question_words,
        matched_terms,
        coverage_ratio
    )


# ============================================================
# Step 5: Cross-Evidence Consistency
# ============================================================

def check_consistency(results, embeddings):

    texts = [
        document.page_content
        for document, score in results
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    similarity_scores = []


    for i in range(len(vectors)):

        for j in range(i + 1, len(vectors)):

            similarity = cosine_similarity(
                [vectors[i]],
                [vectors[j]]
            )[0][0]

            similarity_scores.append(
                similarity
            )


    if len(similarity_scores) > 0:

        average_similarity = (
            sum(similarity_scores)
            / len(similarity_scores)
        )

    else:

        average_similarity = 0


    return (
        average_similarity,
        similarity_scores
    )


# ============================================================
# Step 6: Question
# ============================================================

question = "What are the ACID properties?"


# ============================================================
# Step 7: Retrieve Evidence
# ============================================================

results = vector_store.similarity_search_with_score(
    question,
    k=3
)


# ============================================================
# Step 8: Evidence Gate
# ============================================================

status, best_score, average_score = (
    check_evidence(results)
)


# ============================================================
# Step 9: Evidence Coverage
# ============================================================

important_terms, matched_terms, coverage_ratio = (
    check_coverage(
        question,
        results
    )
)


# ============================================================
# Step 10: Cross-Evidence Consistency
# ============================================================

average_consistency, pairwise_scores = (
    check_consistency(
        results,
        embeddings
    )
)


# ============================================================
# Step 11: Display Evidence Evaluation
# ============================================================

print("Question:", question)


print(
    "\n================ EVIDENCE EVALUATION ================"
)


# ------------------------------------------------------------
# Semantic Relevance
# ------------------------------------------------------------

print("\nSemantic Relevance")
print("------------------")

print(
    "Best distance:",
    round(best_score, 4)
)

print(
    "Average distance:",
    round(average_score, 4)
)

print(
    "Retrieved chunks:",
    len(results)
)


# ------------------------------------------------------------
# Evidence Coverage
# ------------------------------------------------------------

print("\nEvidence Coverage")
print("-----------------")

print(
    "Important question terms:",
    important_terms
)

print(
    "Matched evidence terms:",
    matched_terms
)

print(
    "Coverage ratio:",
    round(coverage_ratio, 4)
)


# ------------------------------------------------------------
# Cross-Evidence Consistency
# ------------------------------------------------------------

print("\nCross-Evidence Consistency")
print("--------------------------")

for i, score in enumerate(
    pairwise_scores
):

    print(
        f"Pair {i + 1} similarity:",
        round(score, 4)
    )


print(
    "Average consistency:",
    round(
        average_consistency,
        4
    )
)


# ------------------------------------------------------------
# Evidence Gate
# ------------------------------------------------------------

print(
    "\nEvidence status:",
    status
)


# ============================================================
# Step 12: Stop if Evidence is Insufficient
# ============================================================

if status == "INSUFFICIENT":

    print(
        "\nEvidence is insufficient."
    )

    print(
        "Generation stopped."
    )


# ============================================================
# Step 13: Continue to Generation
# ============================================================

else:

    print(
        "\nEvidence is sufficient."
    )

    print(
        "Generation can continue."
    )


    # --------------------------------------------------------
    # Build Context
    # --------------------------------------------------------

    context_parts = []


    for i, (document, score) in enumerate(
        results
    ):

        context_parts.append(
            f"""
Evidence {i + 1}
Source: {document.metadata.get("source")}
Page: {document.metadata.get("page", 0) + 1}

{document.page_content}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Step 14: Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are DocLens AI, an evidence-first document assistant.

Answer the user's question using ONLY the provided evidence.

If the evidence does not contain enough information,
do not invent facts.

User Question:
{question}

Evidence:
{context}

Answer:
"""
    )


    # --------------------------------------------------------
    # Step 15: Groq
    # --------------------------------------------------------

    model = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )


    # --------------------------------------------------------
    # Step 16: Chain
    # --------------------------------------------------------

    chain = prompt | model


    # --------------------------------------------------------
    # Step 17: Generate Answer
    # --------------------------------------------------------

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )


    # --------------------------------------------------------
    # Step 18: Display Answer
    # --------------------------------------------------------

    print(
        "\n================ ANSWER ================\n"
    )

    print(
        response.content
    )