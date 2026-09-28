from requirement_evidence import evaluate_question

from evidence_gate import check_evidence

from generator import generate_answer

# Build verified evidence

def build_verified_evidence(evaluation):

    evidence_parts = []

    evidence_items = []

    evidence_number = 1

    for result in evaluation["results"]:

        attribution = result.get(
            "attribution"
        )

        if attribution is None:
            continue

        # Text version for the LLM

        evidence_parts.append(
            f"""
Evidence {evidence_number}

Requirement:
{attribution["requirement"]}

Supporting Evidence:
{attribution["evidence"]}

Source:
{attribution["source"]}

Page:
{attribution["page"]}
"""
        )

        # Structured version for API/frontend

        evidence_items.append({
            "requirement": attribution["requirement"],
            "evidence": attribution["evidence"],
            "source": attribution["source"],
            "page": attribution["page"]
        })

        evidence_number += 1

    return (
        "\n".join(evidence_parts),
        evidence_items
    )

# Run complete RAG pipeline

def run_rag(question, document_id):

    # Step 1: Evaluate question

    evaluation = evaluate_question(
        question,
        document_id,
        k=3
    )

    # Step 2: Evidence Gate

    gate_result = check_evidence(
        evaluation
    )

    # Step 3: Stop if evidence insufficient

    if gate_result["status"] != "SUFFICIENT":

        return {
            "question": question,
            "document_id": document_id,
            "status": "INSUFFICIENT",

            "answer": (
                "The provided document does not contain "
                "enough evidence to answer this question."
            ),

            "evidence": [],

            "gate": {
                "supported_requirements":
                    gate_result["supported_requirements"],

                "total_requirements":
                    gate_result["total_requirements"],

                "coverage":
                    gate_result["coverage"]
            }
        }

    # Step 4: Build verified evidence

    verified_evidence, evidence_items = (
        build_verified_evidence(
            evaluation
        )
    )

    # Step 5: Generate answer

    answer = generate_answer(
        question,
        verified_evidence
    )

    # Step 6: Return API-safe result

    return {
        "question": question,
        "document_id": document_id,
        "status": "SUFFICIENT",
        "answer": answer,
        "evidence": evidence_items,

        "gate": {
            "supported_requirements":
                gate_result["supported_requirements"],

            "total_requirements":
                gate_result["total_requirements"],

            "coverage":
                gate_result["coverage"]
        }
    }

# Test the pipeline directly

if __name__ == "__main__":

    question = "What is a README, and how do I deploy a Kubernetes cluster?"
    document_id = "9a5a4b4a-488e-4a38-a171-07160cbc735b"

    result = run_rag(
        question,
        document_id
    )

    print("QUESTION:")
    print(result["question"])

    print("\nDOCUMENT ID:")
    print(result["document_id"])

    print("\n================ STATUS ================\n")

    print(result["status"])

    print("\n================ ANSWER ================\n")

    print(result["answer"])

    print("\n================ EVIDENCE GATE ================\n")

    print(
        "Supported requirements:",
        result["gate"]["supported_requirements"]
    )

    print(
        "Total requirements:",
        result["gate"]["total_requirements"]
    )

    print(
        "Evidence coverage:",
        result["gate"]["coverage"]
    )

    print(
        "\n================ VERIFIED EVIDENCE ================\n"
    )

    for item in result["evidence"]:

        print("Requirement:")
        print(item["requirement"])

        print("Evidence:")
        print(item["evidence"])

        print("Source:")
        print(item["source"])

        print("Page:")
        print(item["page"])

        print("\n------------------------------")