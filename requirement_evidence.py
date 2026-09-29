
import re

from requirement_extractor import extract_requirements
from requirement_retriever import retrieve_for_requirement
from evidence_verifier import verify_requirement
from evidence_evaluator import evaluate_evidence

# PROJECT-SPECIFIC EVIDENCE ACCEPTANCE CRITERIA

MIN_DIRECTNESS = 1
MIN_COMPLETENESS = 1

# NORMALIZE TEXT FOR EVIDENCE ATTRIBUTION

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

# EVALUATE ONE REQUIREMENT

def evaluate_requirement(requirement, document_id, k=5, question=None):

    # 1. Retrieve candidate evidence chunks
    results = retrieve_for_requirement(
        requirement,
        document_id,
        k=k,
        question=question
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
            evaluation.get("relevance", 0) >= 1
            and (
                evaluation.get("directness", 0) >= MIN_DIRECTNESS
                or evaluation.get("completeness", 0) >= MIN_COMPLETENESS
            )
        ):
            accepted_candidates.append(candidate)

    # If strict check passed nothing, allow top relevant candidate to be verified
    if not accepted_candidates and candidate_evaluations:
        relevant = [c for c in candidate_evaluations if c["evaluation"].get("relevance", 0) >= 1]
        if relevant:
            accepted_candidates.append(min(relevant, key=lambda c: c["distance"]))

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

            # 1. Exact substring match
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

            # 2. Token overlap fallback if verifier paraphrased or truncated
            if attribution is None:
                ev_tokens = set(re.findall(r"\w+", normalized_verified.lower()))
                for candidate in accepted_candidates:
                    document = candidate["document"]
                    cand_tokens = set(re.findall(r"\w+", document.page_content.lower()))
                    if ev_tokens and len(cand_tokens.intersection(ev_tokens)) / len(ev_tokens) >= 0.4:
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

            # 3. Default to top accepted candidate chunk
            if attribution is None and accepted_candidates:
                top_doc = accepted_candidates[0]["document"]
                attribution = {
                    "requirement": requirement,
                    "evidence": verified_evidence,
                    "source": top_doc.metadata.get(
                        "source",
                        "Unknown source"
                    ),
                    "page": top_doc.metadata.get(
                        "page",
                        0
                    ) + 1
                }

    return {
        "requirement": requirement,
        "retrieved_results": results,
        "candidate_evaluations": candidate_evaluations,
        "accepted_candidates": accepted_candidates,
        "verification": verification,
        "attribution": attribution
    }

# PARSE REQUIREMENTS

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

    if not requirements and requirements_text.strip():
        for line in requirements_text.splitlines():
            cleaned = re.sub(r"^[-*•\d\.]+\s*", "", line.strip())
            if cleaned:
                requirements.append(cleaned)

    return requirements

# EVALUATE COMPLETE QUESTION

def evaluate_question(question, document_id, k=5):

    requirements_text = extract_requirements(question)

    requirements = parse_requirements(
        requirements_text
    )

    if not requirements:
        requirements = [question.strip()]

    results = []

    for requirement in requirements:

        result = evaluate_requirement(
            requirement,
            document_id,
            k=k,
            question=question
        )

        results.append(result)

    return {
        "question": question,
        "requirements": requirements,
        "results": results
    }

# CALCULATE REQUIREMENT COVERAGE

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