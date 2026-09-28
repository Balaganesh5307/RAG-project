from requirement_evidence import evaluate_question


question = "What is the purpose of a README?"

document_id = "9a5a4b4a-488e-4a38-a171-07160cbc735b"


evaluation = evaluate_question(
    question,
    document_id,
    k=3
)


print("\n================ REQUIREMENTS ================\n")

print(evaluation["requirements"])


for result in evaluation["results"]:

    print("\n==============================================")
    print("REQUIREMENT:")
    print(result["requirement"])

    print("\nVERIFICATION:")
    print(result["verification"])

    print("\nATTRIBUTION:")
    print(result["attribution"])

    print("\nRETRIEVED RESULTS:")

    for document, score in result["retrieved_results"]:

        print("\nDistance:", score)

        print("Page:", document.metadata.get("page", 0) + 1)

        print("Content:")
        print(document.page_content)