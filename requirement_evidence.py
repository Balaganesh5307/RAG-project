
import re

from requirement_extractor import extract_requirements
from requirement_retriever import retrieve_for_requirement
from evidence_verifier import verify_requirement
from evidence_evaluator import evaluate_evidence


# ============================================================
# PROJECT-SPECIFIC EVIDENCE ACCEPTANCE CRITERIA
# ============================================================

MIN_DIRECTNESS = 2
MIN_COMPLETENESS = 2


# ============================================================
# NORMALIZE TEXT FOR EVIDENCE ATTRIBUTION
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# EVALUATE ONE REQUIREMENT
# ============================================================

def evaluate_requirement(requirement, document_id, k=3):

    # 1. Retrieve candidate evidence chunks
    results = retrieve_for_requirement(
        requirement,
        document_id,
        k=k
    )

    candidate_evaluations = []
    accepted_candidates = []

    # 2. Evaluate each candidate independently
    for document, distance in results:

        evaluation = evaluate_evidence(
            requirement,
            document.page_content
        )

        candidate = {
            "document": document,
            "distance": distance,
            "evaluation": evaluation
        }

        candidate_evaluations.append(candidate)

        # 3. Apply the current project-specific criteria
        if (
            evaluation["directness"] >= MIN_DIRECTNESS
            and
            evaluation["completeness"] >= MIN_COMPLETENESS
        ):
            accepted_candidates.append(candidate)

    # 4. If no candidate passes the criteria, reject the requirement
    if not accepted_candidates:

        verification = {
            "status": "NOT_SUPPORTED",
            "evidence": None,
            "reason": (
                "No retrieved candidate met the current "
                "directness and completeness criteria."
            )
        }

        return {
            "requirement": requirement,
            "retrieved_results": results,
            "candidate_evaluations": candidate_evaluations,
            "accepted_candidates": [],
            "verification": verification,
            "attribution": None
        }

    # 5. Build verifier input from accepted candidates only
    evidence_parts = []

    for index, candidate in enumerate(accepted_candidates):

        document = candidate["document"]
        distance = candidate["distance"]

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            0
        ) + 1

        evidence_parts.append(
            f"""
Evidence {index + 1}
Source: {source}
Page: {page}
Distance: {round(distance, 4)}

{document.page_content}
"""
        )

    evidence_text = "\n".join(evidence_parts)

    # 6. Verify whether accepted evidence supports the requirement
    verification = verify_requirement(
        requirement,
        evidence_text
    )

    # 7. Attribute the verifier's evidence to the original chunk
    attribution = None

    if verification["status"] == "SUPPORTED":

        verified_evidence = verification.get("evidence")

        if verified_evidence:

            normalized_verified = normalize_text(
                verified_evidence
            )

            for candidate in accepted_candidates:

                document = candidate["document"]

                normalized_content = normalize_text(
                    document.page_content
                )

                if normalized_verified in normalized_content:

                    attribution = {
                        "requirement": requirement,
                        "evidence": verified_evidence,
                        "source": document.metadata.get(
                            "source",
                            "Unknown source"
                        ),
                        "page": document.metadata.get(
                            "page",
                            0
                        ) + 1
                    }

                    break

            # Prevent unsupported attribution
            if attribution is None:
                verification = {
                    "status": "NOT_SUPPORTED",
                    "evidence": None,
                    "reason": (
                        "The verifier's quoted evidence could not "
                        "be matched to an accepted source chunk."
                    )
                }

    return {
        "requirement": requirement,
        "retrieved_results": results,
        "candidate_evaluations": candidate_evaluations,
        "accepted_candidates": accepted_candidates,
        "verification": verification,
        "attribution": attribution
    }


# ============================================================
# PARSE REQUIREMENTS
# ============================================================

def parse_requirements(requirements_text):

    requirements = []

    for line in requirements_text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line[0].isdigit() and "." in line:

            requirement = line.split(".", 1)[1].strip()

            if requirement:
                requirements.append(requirement)

    return requirements


# ============================================================
# EVALUATE COMPLETE QUESTION
# ============================================================

def evaluate_question(question, document_id, k=3):

    requirements_text = extract_requirements(question)

    requirements = parse_requirements(
        requirements_text
    )

    results = []

    for requirement in requirements:

        result = evaluate_requirement(
            requirement,
            document_id,
            k=k
        )

        results.append(result)

    return {
        "question": question,
        "requirements": requirements,
        "results": results
    }


# ============================================================
# CALCULATE REQUIREMENT COVERAGE
# ============================================================

def calculate_coverage(results):

    total_requirements = len(results)
    supported_requirements = 0

    for result in results:

        status = result["verification"]["status"]

        if status == "SUPPORTED":
            supported_requirements += 1

    if total_requirements == 0:
        coverage = 0
    else:
        coverage = (
            supported_requirements / total_requirements
        )

    return (
        supported_requirements,
        total_requirements,
        coverage
    )