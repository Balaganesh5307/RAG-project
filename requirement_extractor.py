
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. ENVIRONMENT AND MODEL

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

# 2. REQUIREMENT EXTRACTION PROMPT

prompt = ChatPromptTemplate.from_template(
    """
You are a requirement extractor for DocLens AI,
an evidence-first document question-answering system.

Your task is to convert the user's question into the
smallest complete set of information requirements.

The requirements will be used to retrieve and verify
evidence from a specific document.

IMPORTANT RULES:

1. Preserve the exact topic and context of the question.

2. Never produce vague requirements when the context
   can be included.

   Bad:
   - Title
   - Description
   - Installation

   Better:
   - README title section
   - README description section
   - README installation section

3. For compound questions, identify each distinct
   information need.

4. Do not split a single list-identification task into
   unnecessary separate requirements.

   Example:
   "What are the main sections of a README?"

   This asks for the names of the main README sections.
   It does not necessarily ask for a detailed explanation
   of every section.

5. If the question asks for a definition, create a
   definition requirement.

6. If the question asks for a list, preserve the list's
   subject and purpose in the requirement.

7. Do not add related concepts that the user did not ask for.

8. Do not answer the question.

9. Do not use outside knowledge to add facts or details
   to the requirements.

10. Keep requirements concise, specific, and independently
    verifiable against document evidence.

11. Return ONLY a numbered list.
12. Do not use Markdown code fences.

EXAMPLES:

Question:
What is schema?

Requirements:
1. Definition of database schema

Question:
What are the ACID properties?

Requirements:
1. Atomicity in database transactions
2. Consistency in database transactions
3. Isolation in database transactions
4. Durability in database transactions

Question:
What is a README and what are its main sections?

Requirements:
1. Definition and purpose of a README
2. Names of the main sections in a README

Question:
What are the layers of a neural network?

Requirements:
1. Names of the layers in a neural network

Question:
Explain supervised and unsupervised learning.

Requirements:
1. Explanation of supervised learning
2. Explanation of unsupervised learning

User Question:
{question}

Requirements:
"""
)

chain = prompt | model

# 3. EXTRACT REQUIREMENTS

def extract_requirements(question):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    response = chain.invoke({
        "question": question.strip()
    })

    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Remove first line (```...) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        raw = "\n".join(lines).strip()

    return raw

# 4. STANDALONE TEST

if __name__ == "__main__":

    test_questions = [
        "What is a README and what are its main sections?",
        "What are the ACID properties?",
        "What is the purpose of a README?"
    ]

    for question in test_questions:

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(question)

        print("\nEXTRACTED REQUIREMENTS:")

        try:
            requirements = extract_requirements(question)
            print(requirements)

        except Exception as error:
            print("Extraction failed:", error)