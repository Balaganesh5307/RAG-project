from requirement_retriever import (
    retrieve_for_requirement
)


document_id = "3dda93c0-5972-43f9-aecb-dd1deac623de"

requirement = "Atomicity"


results = retrieve_for_requirement(
    requirement,
    document_id,
    k=3
)


print("Requirement:")
print(requirement)

print("\nDocument ID:")
print(document_id)

print("\nRetrieved Results:")


for i, (document, score) in enumerate(results):

    print("\n==============================")
    print(f"RESULT {i + 1}")
    print("==============================")

    print("Distance:")
    print(score)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)