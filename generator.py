from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env
load_dotenv()


# Create the Groq language model
model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


# Prompt used for answer generation
prompt = ChatPromptTemplate.from_template(
    """
You are DocLens AI, an evidence-first document assistant.

Answer the user's question using ONLY the provided evidence.

Rules:

1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Answer the user's question directly.
5. Preserve the meaning of the provided evidence.
6. Keep the answer clear and concise.
7. Do not mention the internal retrieval or verification process.
8. If the provided evidence does not contain enough information,
   say that the information is not available in the provided evidence.

User Question:
{question}

Verified Evidence:
{evidence}

Answer:
"""
)


# Connect the prompt to the Groq model
chain = prompt | model


def generate_answer(question, evidence):

    response = chain.invoke(
        {
            "question": question,
            "evidence": evidence
        }
    )

    return response.content


# Test the generator when this file is run directly
if __name__ == "__main__":

    question = "What are the ACID properties?"

    evidence = """
Atomicity: Transactions are all-or-nothing.

Consistency: Database moves from one valid state to another.

Isolation: Concurrent transactions don’t interfere.

Durability: Once committed, data persists even after failures.
"""

    answer = generate_answer(
        question,
        evidence
    )

    print("QUESTION:")
    print(question)

    print("\n================ ANSWER ================\n")

    print(answer)
