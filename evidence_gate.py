from requirement_evidence import (
    calculate_coverage
)


def check_evidence(evaluation):

    results = evaluation["results"]

    supported_requirements, total_requirements, coverage = (
        calculate_coverage(results)
    )

    distances = []

    for result in results:

        for document, score in result["retrieved_results"]:

            distances.append(score)

    if distances:

        best_distance = min(distances)

        average_distance = (
            sum(distances) /
            len(distances)
        )

    else:

        best_distance = None
        average_distance = None

    if total_requirements == 0 or supported_requirements == 0:

        status = "INSUFFICIENT"

    elif coverage >= 0.5 or supported_requirements >= 1:

        status = "SUFFICIENT"

    else:

        status = "INSUFFICIENT"

    return {
        "status": status,
        "supported_requirements": supported_requirements,
        "total_requirements": total_requirements,
        "coverage": coverage,
        "best_distance": best_distance,
        "average_distance": average_distance
    }