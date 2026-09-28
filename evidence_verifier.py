import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
You are an evidence verifier for an evidence-first
document question-answering system.

Your task is to determine whether the provided evidence
supports the given information requirement.

Rules:

1. Use ONLY the provided evidence.
2. Do not use outside knowledge.
3. Do not assume information that is not explicitly supported.
4. If the evidence directly supports the requirement,
   return SUPPORTED.
5. If the evidence does not support the requirement,
   return NOT_SUPPORTED.
6. If supported, quote the exact supporting text from the evidence.
7. If not supported, evidence must be None.
8. Keep the reason concise.
9. Do not modify or invent evidence.
10. Return ONLY valid JSON.
11. Do not use markdown code fences.

Information Requirement:
{requirement}

Evidence:
{evidence}

Return exactly this JSON structure:

{{
    "status": "SUPPORTED" or "NOT_SUPPORTED",
    "evidence": "exact supporting text" or null,
    "reason": "short explanation"
}}
"""
)


chain = prompt | model


def verify_requirement(requirement, evidence):

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

    result = json.loads(raw)

    return result