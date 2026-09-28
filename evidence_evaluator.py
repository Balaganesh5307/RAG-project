import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

# 2. INITIALIZE LLM

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

# 3. EVIDENCE EVALUATION PROMPT

prompt = ChatPromptTemplate.from_template(
    """
You are an evidence quality evaluator for DocLens AI,
an evidence-first document question-answering system.

Your task is to evaluate how well a candidate evidence
passage supports a specific information requirement.

Use ONLY the provided requirement and evidence.

Evaluate the following three dimensions.

1. RELEVANCE
Does the evidence discuss the topic or information
requested by the requirement?

2. DIRECTNESS
Does the evidence directly state or explain the
required information, rather than merely mentioning it?

3. COMPLETENESS
Does the evidence contain enough information to
satisfy the requirement?

Use integer scores from 0 to 3 for each dimension.

SCORING GUIDE:

0 = No meaningful support.
1 = Weak support or indirect mention.
2 = Moderate support; some required information is present.
3 = Strong, direct, and sufficiently complete support.

IMPORTANT RULES:

- Use only the provided evidence.
- Do not use outside knowledge.
- Do not assume missing information.
- Do not invent facts.
- A mention of a concept is not automatically direct evidence.
- Evaluate evidence against the exact requirement.
- If evidence only mentions a section without explaining it,
  do not give it a high directness or completeness score.
- If evidence is incomplete, reflect that in the score.
- Keep the reason concise and evidence-specific.
- Return ONLY valid JSON.
- Do not use Markdown code fences.
- Do not include additional keys.

Information Requirement:
{requirement}

Candidate Evidence:
{evidence}

Return exactly this JSON structure:

{{
    "relevance": 0,
    "directness": 0,
    "completeness": 0,
    "reason": "Short explanation of the evaluation"
}}
"""
)

chain = prompt | model

# 4. VALIDATE EVALUATION RESULT

def validate_evaluation(result):
    """
    Validate the structure and values returned by the LLM.

    Each evaluation score must be an integer from 0 to 3.
    """

    required_keys = {
        "relevance",
        "directness",
        "completeness",
        "reason"
    }

    if not isinstance(result, dict):
        raise ValueError("Evaluation result must be a JSON object.")

    if set(result.keys()) != required_keys:
        raise ValueError(
            "Evaluation result contains missing or unexpected keys."
        )

    score_keys = [
        "relevance",
        "directness",
        "completeness"
    ]

    for key in score_keys:
        score = result[key]

        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(
                f"{key} must be an integer."
            )

        if score < 0 or score > 3:
            raise ValueError(
                f"{key} must be between 0 and 3."
            )

    if not isinstance(result["reason"], str):
        raise ValueError("Reason must be a string.")

    return result

# 5. EVALUATE ONE CANDIDATE EVIDENCE

def evaluate_evidence(requirement, evidence):
    """
    Evaluate one evidence passage against one requirement.

    Args:
        requirement (str): Information the evidence must support.
        evidence (str): Candidate evidence retrieved from the document.

    Returns:
        dict: Relevance, directness, completeness, and reason.
    """

    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("Requirement cannot be empty.")

    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Evidence cannot be empty.")

    response = chain.invoke({
        "requirement": requirement,
        "evidence": evidence
    })

    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The LLM returned invalid JSON."
        ) from error

    return validate_evaluation(result)

# 6. COMPARE WEAK AND DIRECT EVIDENCE

if __name__ == "__main__":

    requirement = "Installation section"

    test_cases = [
        {
            "name": "Weak Evidence",
            "evidence": """
            3. Ask a classmate to follow only your Installation
            section on a fresh machine, and fix anything that
            confused them.
            """
        },
        {
            "name": "Direct Evidence",
            "evidence": """
            ## Installation

            Instructions for setting up the project on a
            fresh machine.
            """
        }
    ]

    print("\n========================================")
    print("DOCLENS AI — EVIDENCE COMPARISON")
    print("========================================")

    for test_case in test_cases:

        print("\n----------------------------------------")
        print("Test:", test_case["name"])
        print("----------------------------------------")

        print("\nRequirement:")
        print(requirement)

        print("\nEvidence:")
        print(test_case["evidence"].strip())

        try:
            result = evaluate_evidence(
                requirement,
                test_case["evidence"]
            )

            print("\nEvaluation:")
            print(json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            ))

            print("\nScore Summary:")
            print("Relevance:", result["relevance"], "/ 3")
            print("Directness:", result["directness"], "/ 3")
            print("Completeness:", result["completeness"], "/ 3")

        except Exception as error:
            print("\nEvaluation failed:")
            print(error)