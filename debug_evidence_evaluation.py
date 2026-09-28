
from requirement_evidence import evaluate_question


QUESTION = "What is a README and what are its main sections?"

DOCUMENT_ID = "9a5a4b4a-488e-4a38-a171-07160cbc735b"


def print_separator():
    print("\n" + "=" * 75)


def main():

    evaluation = evaluate_question(
        question=QUESTION,
        document_id=DOCUMENT_ID,
        k=3
    )

    print_separator()
    print("DOCLENS AI — EVIDENCE EVALUATION DIAGNOSTIC")
    print_separator()

    print("\nQuestion:")
    print(evaluation["question"])

    print("\nExtracted Requirements:")

    for index, requirement in enumerate(
        evaluation["requirements"],
        start=1
    ):
        print(f"{index}. {requirement}")

    for result_index, result in enumerate(
        evaluation["results"],
        start=1
    ):

        print_separator()

        print(f"REQUIREMENT {result_index}")
        print_separator()

        print("\nRequirement:")
        print(result["requirement"])

        print("\nVerification:")
        print(result["verification"])

        attribution = result.get("attribution")

        print("\nAttribution:")
        print(attribution)

        candidates = result.get(
            "candidate_evaluations",
            []
        )

        print("\nCandidate Evaluations:")

        if not candidates:
            print("No candidates retrieved.")
            continue

        accepted_count = 0

        for candidate_index, candidate in enumerate(
            candidates,
            start=1
        ):

            document = candidate["document"]
            distance = candidate["distance"]
            scores = candidate["evaluation"]

            accepted = (
                scores["directness"] >= 2
                and scores["completeness"] >= 2
            )

            if accepted:
                accepted_count += 1

            print("\n------------------------------")
            print(f"CANDIDATE {candidate_index}")
            print("------------------------------")

            print("Page:", document.metadata.get("page", 0) + 1)
            print("Distance:", round(distance, 4))

            print("\nRelevance:", scores["relevance"], "/ 3")
            print("Directness:", scores["directness"], "/ 3")
            print("Completeness:", scores["completeness"], "/ 3")

            print("Accepted:", accepted)

            print("\nReason:")
            print(scores["reason"])

            print("\nContent:")
            print(document.page_content)

        print("\nAccepted Candidates:", accepted_count, "/", len(candidates))

        print("\nFinal Requirement Status:")
        print(result["verification"]["status"])

    print_separator()
    print("DIAGNOSTIC COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()